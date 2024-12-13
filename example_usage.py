import asyncio
from autoscan.autoscan import autoscan

async def main():
    result = await autoscan(pdf_path="examples/helloworld.pdf")

# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())