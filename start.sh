#!/bin/bash

# Install dependencies
pip3 install -r requirements.txt

# Install ffmpeg
apt-get update && apt-get install -y ffmpeg

# Run the bot
python3 bot.py