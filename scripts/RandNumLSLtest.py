import random
import time
from pylsl import StreamInfo, StreamOutlet, local_clock

num_stream = StreamInfo(name="RandNum", type="RandNum", channel_count=1,
                             channel_format='double64', source_id='randnum')
outlet = StreamOutlet(num_stream)


# Calculate the end time
end_time = time.time() + 300  # 5 minutes = 300 seconds

while time.time() < end_time:
    # Generate a random value between 1 and 10
    random_value = random.randint(1, 10)
    outlet.push_sample([random_value], timestamp=local_clock())
    # Print the random value
    print(random_value)

    # Wait for 1 second
    time.sleep(.1)