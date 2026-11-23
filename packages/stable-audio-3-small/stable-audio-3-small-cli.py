#!/usr/bin/env python3
import os
import sys

CANONICAL = "/home/ubuntu/opencode/cli-ai/packages/stable-audio-3/stable-audio-3-cli.py"
os.execv(sys.executable, [sys.executable, CANONICAL] + sys.argv[1:])