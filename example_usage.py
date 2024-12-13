import asyncio
from autoscan.autoscan import autoscan

async def main():
    result = await autoscan(pdf_path="examples/helloworld.pdf", temp_dir="examples/temp", cleanup_temp=False)

# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())