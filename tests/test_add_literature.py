"""Test AddLiteratureService - run directly to see service execution trajectory."""

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.services.add_literature import AddLiteratureService
from pangu_agent.tools import (
    ExploreLibraryTool,
    ViewFileTool,
    MoveFileTool,
    SearchLibraryTool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_LIBRARY_ROOT = Path(__file__).parent.parent / "runs" / "test_add_literature_library"
TEST_SOURCE_DIR = Path(__file__).parent.parent / "runs" / "test_source_files"


def setup_test_environment():
    """Reset and create a fresh test library and source directory."""
    # Clean up existing directories
    for directory in [TEST_LIBRARY_ROOT, TEST_SOURCE_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
            print(f"✓ Removed existing directory: {directory.name}")

    # Create test library structure
    TEST_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    (TEST_LIBRARY_ROOT / ".inbox").mkdir(exist_ok=True)
    print(f"✓ Created test library at: {TEST_LIBRARY_ROOT}")

    # Create source directory with test files
    TEST_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (TEST_SOURCE_DIR / "subdir").mkdir(exist_ok=True)
    print(f"✓ Created source directory at: {TEST_SOURCE_DIR}")

    # Download real PDF files from arXiv
    pdf_url_1 = "https://arxiv.org/pdf/1706.03762.pdf"
    test_pdf = TEST_SOURCE_DIR / "attention_paper.pdf"
    print(f"  Downloading PDF from {pdf_url_1}...")
    try:
        urlretrieve(pdf_url_1, test_pdf)
        print(f"  ✓ Downloaded attention_paper.pdf")
    except Exception as e:
        print(f"  ✗ Failed to download attention_paper.pdf: {e}")
        raise RuntimeError(f"Failed to download test PDF: {e}")

    # Download a real image
    img_url = "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"
    test_img = TEST_SOURCE_DIR / "test_figure.png"
    print(f"  Downloading image from {img_url}...")
    try:
        urlretrieve(img_url, test_img)
        print(f"  ✓ Downloaded test_figure.png")
    except Exception as e:
        print(f"  ✗ Failed to download image: {e}")
        raise RuntimeError(f"Failed to download test image: {e}")

    # Download another PDF for subdirectory
    pdf_url_2 = "https://arxiv.org/pdf/1409.1556.pdf"  # VGG paper
    nested_pdf = TEST_SOURCE_DIR / "subdir" / "nested_paper.pdf"
    print(f"  Downloading nested PDF from {pdf_url_2}...")
    try:
        urlretrieve(pdf_url_2, nested_pdf)
        print(f"  ✓ Downloaded subdir/nested_paper.pdf")
    except Exception as e:
        print(f"  ✗ Failed to download nested_paper.pdf: {e}")
        raise RuntimeError(f"Failed to download nested PDF: {e}")

    # Create non-addable files (should be ignored)
    (TEST_SOURCE_DIR / "readme.txt").write_text("This should be ignored")
    (TEST_SOURCE_DIR / "data.csv").write_text("x,y\n1,2\n")
    print(f"  ✓ Created non-addable files (for testing filtering)")

    print(f"✓ Test environment setup complete")


def test_scan_addable_files():
    """Test scanning for addable files (PDFs and images)."""
    print("\n" + "=" * 60)
    print("TEST: Scan Addable Files")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
    ]

    service = AddLiteratureService(manager, llm_client, tools)

    # Test 1: Scan directory
    print("\n[Test 1] Scan directory for addable files:")
    files = service.scan_addable_files(str(TEST_SOURCE_DIR))
    print(f"  Found {len(files)} addable file(s):")
    for file in files:
        print(f"    - {file.name}")

    assert len(files) == 3, f"Expected 3 addable files, found {len(files)}"
    file_names = {f.name for f in files}
    expected = {"attention_paper.pdf", "test_figure.png", "nested_paper.pdf"}
    assert file_names == expected, f"File names mismatch: {file_names} vs {expected}"
    print("  ✓ Correctly found 3 addable files (2 PDFs + 1 PNG)")

    # Test 2: Scan single file
    print("\n[Test 2] Scan single file:")
    single_file = service.scan_addable_files(str(TEST_SOURCE_DIR / "attention_paper.pdf"))
    print(f"  Found {len(single_file)} file(s):")
    for file in single_file:
        print(f"    - {file.name}")

    assert len(single_file) == 1, f"Expected 1 file, found {len(single_file)}"
    assert single_file[0].name == "attention_paper.pdf"
    print("  ✓ Correctly scanned single file")

    # Test 3: Non-existent path
    print("\n[Test 3] Non-existent path (should raise error):")
    try:
        service.scan_addable_files("/nonexistent/path")
        print("  ✗ Should have raised FileNotFoundError")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  ✓ Correctly raised FileNotFoundError: {e}")

    print("\n✓ All scan_addable_files tests passed!")


def test_add_single_file():
    """Test adding a single file with LLM organization."""
    print("\n" + "=" * 60)
    print("TEST: Add Single File with LLM")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
    ]

    service = AddLiteratureService(manager, llm_client, tools)

    test_file = str(TEST_SOURCE_DIR / "attention_paper.pdf")
    print(f"\n[Adding file]: {test_file}")

    result = service.add_file_with_llm(test_file)

    print(f"\n[Result]:")
    print(f"  Success: {result['success']}")
    print(f"  Source: {result['source']}")
    if result['success']:
        print(f"  Destination: {result['destination']}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")

    # Verify the file was moved successfully
    if result['success']:
        dest_path = TEST_LIBRARY_ROOT / result['destination']
        assert dest_path.exists(), f"File not found at destination: {dest_path}"
        print(f"  ✓ File exists at destination")

        # Verify it's no longer in inbox
        inbox_files = list((TEST_LIBRARY_ROOT / ".inbox").glob("*.pdf"))
        print(f"  Files in inbox: {len(inbox_files)}")
        assert len(inbox_files) == 0, "File should have been moved out of inbox"
        print(f"  ✓ File moved out of inbox")
    else:
        print(f"  ✗ Failed to add file: {result.get('error')}")

    print("\n✓ Single file add test completed!")


def test_add_directory():
    """Test adding multiple files from a directory."""
    print("\n" + "=" * 60)
    print("TEST: Add Directory (Multiple Files)")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
        SearchLibraryTool(manager),
    ]

    service = AddLiteratureService(manager, llm_client, tools)

    print(f"\n[Adding directory]: {TEST_SOURCE_DIR}")

    results = service.add_path(str(TEST_SOURCE_DIR))

    print(f"\n[Results Summary]:")
    print(f"  Total files processed: {len(results)}")

    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count

    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")

    print(f"\n[Details]:")
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"  [{i}] {status} {Path(result['source']).name}")
        if result['success']:
            print(f"       → {result['destination']}")
        else:
            print(f"       Error: {result.get('error', 'Unknown')}")

    # Verify expected number of files
    assert len(results) == 3, f"Expected 3 files, processed {len(results)}"

    print("\n✓ Directory add test completed!")


def test_empty_directory():
    """Test adding from an empty directory."""
    print("\n" + "=" * 60)
    print("TEST: Add Empty Directory")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
    ]

    service = AddLiteratureService(manager, llm_client, tools)

    # Create empty directory
    empty_dir = TEST_SOURCE_DIR / "empty_subdir"
    empty_dir.mkdir(exist_ok=True)

    print(f"\n[Adding empty directory]: {empty_dir}")

    results = service.add_path(str(empty_dir))

    print(f"\n[Result]:")
    print(f"  Files processed: {len(results)}")

    assert len(results) == 0, f"Expected 0 files, got {len(results)}"
    print("  ✓ Correctly returned empty results for empty directory")

    print("\n✓ Empty directory test passed!")


def main():
    parser = argparse.ArgumentParser(description="Test AddLiteratureService with configurable selection")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["scan", "single", "directory", "empty", "all"],
        default=["all"],
        help="Select which tests to run (default: all)",
    )
    args = parser.parse_args()

    print("Setting up test environment...")
    setup_test_environment()

    # Determine which tests to run
    run_all = "all" in args.tests
    tests_to_run = args.tests if not run_all else ["scan", "single", "directory", "empty"]

    # Run selected tests
    if "scan" in tests_to_run:
        test_scan_addable_files()

    if "single" in tests_to_run:
        # Reset library for single file test
        if (TEST_LIBRARY_ROOT / ".inbox").exists():
            shutil.rmtree(TEST_LIBRARY_ROOT / ".inbox")
        (TEST_LIBRARY_ROOT / ".inbox").mkdir()
        test_add_single_file()

    if "directory" in tests_to_run:
        # Reset library for directory test
        if TEST_LIBRARY_ROOT.exists():
            shutil.rmtree(TEST_LIBRARY_ROOT)
        TEST_LIBRARY_ROOT.mkdir(parents=True)
        (TEST_LIBRARY_ROOT / ".inbox").mkdir()
        test_add_directory()

    if "empty" in tests_to_run:
        test_empty_directory()

    print("\n" + "=" * 60)
    print("All AddLiteratureService tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
