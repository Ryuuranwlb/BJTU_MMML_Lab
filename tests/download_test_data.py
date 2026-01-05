#!/usr/bin/env python3
"""
Download test data script
Downloads PDF and image files to a specified folder for testing
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Dict, Any


def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file to the specified path

    Args:
        url: File URL
        dest_path: Destination path

    Returns:
        bool: Whether download succeeded
    """
    try:
        print(f"Downloading: {url}")
        print(f"Saving to: {dest_path}")

        # Set User-Agent to avoid being blocked by some servers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response:
            with open(dest_path, 'wb') as out_file:
                out_file.write(response.read())

        print(f"✓ Downloaded successfully: {dest_path.name}\n")
        return True
    except Exception as e:
        print(f"✗ Download failed {dest_path.name}: {str(e)}\n")
        return False


def main():
    """Main function"""
    # Set default download directory to tests/test_data
    default_dir = Path(__file__).parent / "test_data"
    subdir = default_dir / "subdir"

    # Create directories
    default_dir.mkdir(exist_ok=True)
    subdir.mkdir(exist_ok=True)
    print(f"Test data directory: {default_dir}")
    print(f"Subdirectory: {subdir}\n")

    # Define files to download
    files_to_download: List[Dict[str, Any]] = [
        # PDF files - Research papers
        {
            "url": "https://arxiv.org/pdf/1706.03762.pdf",  # Attention is All You Need
            "filename": "attention_is_all_you_need.pdf",
            "type": "pdf",
            "subdir": False
        },
        {
            "url": "https://arxiv.org/pdf/2010.11929.pdf",  # Vision Transformer
            "filename": "vision_transformer.pdf",
            "type": "pdf",
            "subdir": False
        },
        {
            "url": "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
            "filename": "bert.pdf",
            "type": "pdf",
            "subdir": True  # This PDF goes to subdir
        },

        # Image files - Public test images
        {
            "url": "https://picsum.photos/800/600",  # Random image
            "filename": "test_image_1.jpg",
            "type": "image",
            "subdir": False
        },
        {
            "url": "https://picsum.photos/1024/768",  # Random image
            "filename": "test_image_2.jpg",
            "type": "image",
            "subdir": False
        },
        {
            "url": "https://picsum.photos/640/480",  # Random image
            "filename": "test_image_3.jpg",
            "type": "image",
            "subdir": True  # This image goes to subdir
        },
    ]

    # Download statistics
    success_count = 0
    fail_count = 0

    # Execute downloads
    print(f"Starting download of {len(files_to_download)} files...\n")
    print("=" * 60)

    for item in files_to_download:
        url = item["url"]
        filename = item["filename"]
        file_type = item["type"]
        use_subdir = item.get("subdir", False)

        # Determine destination path
        target_dir = subdir if use_subdir else default_dir
        dest_path = target_dir / filename
        location = "subdir" if use_subdir else "root"

        print(f"[{file_type.upper()}] {filename} -> {location}")
        if download_file(url, dest_path):
            success_count += 1
        else:
            fail_count += 1

    # Print statistics
    print("=" * 60)
    print(f"\nDownload complete!")
    print(f"Success: {success_count} files")
    print(f"Failed: {fail_count} files")
    print(f"\nAll files saved in: {default_dir}")

    # List downloaded files
    if success_count > 0:
        print("\nDownloaded files:")
        print("\nRoot directory:")
        for file in sorted(default_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")

        if subdir.exists() and any(subdir.iterdir()):
            print("\nSubdirectory:")
            for file in sorted(subdir.iterdir()):
                if file.is_file():
                    size = file.stat().st_size
                    size_mb = size / (1024 * 1024)
                    print(f"  - {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
