# Gameplay annotations for the shinobi dataset

In order to benefit from the complex structure of video game tasks, a number of variables are extracted from the replays obtained during data acquisitions. The produced annotations are encoded in a BIDS-compatible format, i.e. a .tsv file with at least 3 rows : onset, duration and event_type (sometimes named trial_type). These files encode the beginning and start of each repetition, but also contains the values of state variables as well as some handcrafted annotations.

## State variables
The values of the variables retrieved from the emulator RAM are encoded in separate columns with a sampling frequency of 60Hz, corresponding to the game frames (displayed at 60fps). In the annotated events file, each frame is encoded as a separate event.
- X_player :
- etc...

## Handcrafted annotations
Some annotations are handcrafted from a combination or a processing of state variables. These events can have varying durations (such as keypresses) or an arbitrary duration determined in generate_annotations.py (default = 0.1sec). 
- Keypresses : RIGHT, LEFT, UP, DOWN, B, C
- Kill :
- HealthLoss : 