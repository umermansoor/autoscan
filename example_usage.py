import asyncio
from autoscan.autoscan import autoscan

import logging

# Configure the logging system in the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def main():
    # await autoscan(pdf_path="https://www.cs.cmu.edu/~ab/15-111N09/Lectures/Lecture%2001%20Introduction.pdf")
    await autoscan(pdf_path="examples/helloworld2.pdf")

# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())