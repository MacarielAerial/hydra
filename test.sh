#!/bin/bash -e

echo "Testing..."
coverage run -m pytest && coverage report -m
