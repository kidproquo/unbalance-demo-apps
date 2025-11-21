"""
Redis Client for Synchronized Window Processing

Provides publisher and consumer classes for coordinating window processing
across multiple unbalance detection approaches using Redis Streams.
"""

import redis
import json
import time
import os
import base64
import numpy as np
from typing import Dict, Optional, List, Union
from datetime import datetime, timezone


def encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for Redis storage."""
    return base64.b64encode(arr.tobytes()).decode('ascii')


def decode_array(data: str, dtype: np.dtype = np.float64, shape: tuple = None) -> np.ndarray:
    """Decode base64 string back to numpy array."""
    arr = np.frombuffer(base64.b64decode(data), dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


class RedisConfig:
    """Redis connection configuration."""

    def __init__(self, host: str = None, port: int = None, db: int = 0):
        """
        Initialize Redis configuration.

        Automatically checks environment variables for Docker deployment:
        - REDIS_HOST (default: localhost)
        - REDIS_PORT (default: 6379)
        - REDIS_DB (default: 0)

        Args:
            host: Redis host (overrides env var)
            port: Redis port (overrides env var)
            db: Redis database number
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db or int(os.getenv('REDIS_DB', 0))

    def create_client(self) -> redis.Redis:
        """Create and return a Redis client."""
        return redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,  # Return strings instead of bytes
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )


class WindowPublisher:
    """
    Publisher for window selections.

    Publishes window selections to a Redis Stream for consumption by
    multiple detection approaches.
    """

    def __init__(self, config: RedisConfig, stream_name: str = 'windows'):
        """
        Initialize window publisher.

        Args:
            config: Redis configuration
            stream_name: Name of the Redis stream to publish to
        """
        self.config = config
        self.stream_name = stream_name
        self.client = None
        self._connect()

    def _connect(self):
        """Establish connection to Redis with retry logic."""
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.client = self.config.create_client()
                self.client.ping()
                print(f"✓ Connected to Redis at {self.config.host}:{self.config.port}")
                return
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Redis connection failed (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts: {e}")

    def publish_window(self, dataset: str, window_idx: int,
                      start_idx: int, end_idx: int,
                      sensor_data: np.ndarray = None) -> str:
        """
        Publish a window selection to the stream.

        Args:
            dataset: Dataset label (e.g., '0E', '3E')
            window_idx: Window index within the selected dataset
            start_idx: Start row index in the original CSV
            end_idx: End row index in the original CSV
            sensor_data: Optional numpy array of sensor data (shape: n_samples x n_columns)
                        Columns: [RPM, Vibration_1, Vibration_2, Vibration_3]

        Returns:
            Message ID from Redis
        """
        message = {
            'dataset': dataset,
            'window_idx': window_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Include sensor data if provided
        if sensor_data is not None:
            message['sensor_data'] = encode_array(sensor_data)
            message['sensor_shape'] = json.dumps(sensor_data.shape)
            message['sensor_dtype'] = str(sensor_data.dtype)

        # Publish to Redis stream using XADD
        message_id = self.client.xadd(
            self.stream_name,
            message,
            maxlen=10000  # Keep last 10000 messages
        )

        return message_id

    def get_stream_length(self) -> int:
        """Get current length of the stream."""
        return self.client.xlen(self.stream_name)

    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()


class WindowConsumer:
    """
    Consumer for window selections.

    Consumes window selections from Redis Stream with consumer group support
    for coordinated processing across multiple approaches.
    """

    def __init__(self, config: RedisConfig, stream_name: str = 'windows',
                 consumer_group: str = 'detectors', consumer_name: str = None):
        """
        Initialize window consumer.

        Args:
            config: Redis configuration
            stream_name: Name of the Redis stream to consume from
            consumer_group: Consumer group name (all approaches use same group)
            consumer_name: Unique consumer name (e.g., 'cnn', 'fft', 'rfc')
        """
        self.config = config
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f'consumer_{os.getpid()}'
        self.client = None
        self._connect()
        self._create_consumer_group()

    def _connect(self):
        """Establish connection to Redis with retry logic."""
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.client = self.config.create_client()
                self.client.ping()
                print(f"✓ Connected to Redis at {self.config.host}:{self.config.port}")
                return
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Redis connection failed (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts: {e}")

    def _create_consumer_group(self):
        """
        Create consumer group if it doesn't exist.

        Each approach should use its own consumer group for fan-out pattern
        (each message delivered to all approaches). If multiple consumers
        share the same group, messages are distributed among them instead.
        """
        try:
            # Try to create consumer group starting from beginning of stream
            self.client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id='0',  # Start from beginning
                mkstream=True  # Create stream if it doesn't exist
            )
            print(f"✓ Created consumer group '{self.consumer_group}'")
        except redis.ResponseError as e:
            if 'BUSYGROUP' in str(e):
                # Group already exists, that's fine
                print(f"✓ Using existing consumer group '{self.consumer_group}'")
            else:
                raise

    def read_window(self, block_ms: int = 5000, count: int = 1) -> Optional[Dict]:
        """
        Read next window selection from stream (blocking).

        Args:
            block_ms: How long to block waiting for new messages (milliseconds)
            count: Maximum number of messages to read

        Returns:
            Dictionary with window selection data, or None if timeout
            Format: {
                'message_id': '1234567890-0',
                'dataset': '3E',
                'window_idx': 42,
                'start_idx': 614400,
                'end_idx': 655360,
                'timestamp': '2025-11-20T17:30:45Z'
            }
        """
        try:
            # Read from consumer group (each consumer gets all messages)
            messages = self.client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: '>'},  # '>' means new messages
                count=count,
                block=block_ms
            )

            if not messages:
                return None  # Timeout

            # Extract message data
            stream_name, message_list = messages[0]
            if not message_list:
                return None

            message_id, data = message_list[0]

            # Parse and return
            result = {
                'message_id': message_id,
                'dataset': data['dataset'],
                'window_idx': int(data['window_idx']),
                'start_idx': int(data['start_idx']),
                'end_idx': int(data['end_idx']),
                'timestamp': data['timestamp']
            }

            # Decode sensor data if present
            if 'sensor_data' in data:
                shape = tuple(json.loads(data['sensor_shape']))
                dtype = np.dtype(data['sensor_dtype'])
                result['sensor_data'] = decode_array(data['sensor_data'], dtype, shape)

            return result

        except redis.ConnectionError as e:
            print(f"⚠️  Redis connection error: {e}")
            time.sleep(1)
            self._connect()  # Reconnect
            return None

    def acknowledge(self, message_id: str):
        """
        Acknowledge that a message has been processed.

        Args:
            message_id: Message ID to acknowledge
        """
        self.client.xack(self.stream_name, self.consumer_group, message_id)

    def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) messages for this consumer."""
        pending = self.client.xpending(self.stream_name, self.consumer_group)
        return pending['pending']

    def wait_for_stream(self, timeout_s: int = 30) -> bool:
        """
        Wait for stream to be created (by coordinator).

        Args:
            timeout_s: Maximum time to wait in seconds

        Returns:
            True if stream exists, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            if self.client.exists(self.stream_name):
                print(f"✓ Stream '{self.stream_name}' is ready")
                return True
            print(f"⏳ Waiting for stream '{self.stream_name}' to be created...")
            time.sleep(1)

        print(f"⚠️  Timeout waiting for stream '{self.stream_name}'")
        return False

    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()


def test_redis_connection(config: RedisConfig) -> bool:
    """
    Test Redis connection.

    Args:
        config: Redis configuration to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = config.create_client()
        client.ping()
        print(f"✓ Redis connection successful: {config.host}:{config.port}")
        client.close()
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test Redis connection
    print("Testing Redis connection...")
    config = RedisConfig()

    if test_redis_connection(config):
        print("\nRedis client ready for use!")
        print(f"  Host: {config.host}")
        print(f"  Port: {config.port}")
        print(f"  DB: {config.db}")
    else:
        print("\nMake sure Redis is running:")
        print("  docker run -p 6379:6379 redis:7-alpine")
