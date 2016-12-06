#!/bin/bash -e
# This script is invoked to uninstall the hcRNG library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# Remove build
rm -rf $current_work_dir/build
