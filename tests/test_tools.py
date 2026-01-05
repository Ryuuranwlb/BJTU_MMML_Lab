"""Test tools - run directly to see outputs."""

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools.base import ToolExecutor
from pangu_agent.tools.explore_library import ExploreLibraryTool
from pangu_agent.tools.move_file import MoveFileTool
from pangu_agent.tools.view_file import ViewFileTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_LIBRARY_ROOT = Path(__file__).parent.parent / "runs" / "test_library"


def print_result(result):
    """Print tool execution result in a unified format."""
    if result.success:
        print(f"✓ Success")
        print(f"Output: {result.output}")
    else:
        print(f"✗ Failed")
        print(f"Error: {result.error}")


def setup_test_library():
    """Reset and create a fresh test library with sample structure."""
    # Remove existing library
    if TEST_LIBRARY_ROOT.exists():
        shutil.rmtree(TEST_LIBRARY_ROOT)
        print(f"✓ Removed existing test library")

    # Create fresh structure
    TEST_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    (TEST_LIBRARY_ROOT / ".inbox").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "papers").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "papers" / "AI").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "books").mkdir(exist_ok=True)

    # Download real PDF - Attention Is All You Need paper
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    test_pdf = TEST_LIBRARY_ROOT / "papers" / "AI" / "transformer.pdf"

    print(f"  Downloading PDF from {pdf_url}...")
    try:
        urlretrieve(pdf_url, test_pdf)
        print(f"  ✓ Downloaded transformer.pdf")
    except Exception as e:
        print(f"  ✗ Failed to download PDF: {e}")
        # Fallback to dummy content
        test_pdf.write_text("attention is all you need")

    # Create metadata file for the PDF
    test_meta = TEST_LIBRARY_ROOT / "papers" / "AI" / ".transformer.pdf.meta.json"
    test_meta.write_text('{"title": "Attention Is All You Need", "year": 2017, "authors": ["Vaswani et al."]}')

    # Download real image - use a reliable public source
    img_url = "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"
    test_img = TEST_LIBRARY_ROOT / "papers" / "AI" / "sample_image.png"

    print(f"  Downloading image from {img_url}...")
    try:
        urlretrieve(img_url, test_img)
        print(f"  ✓ Downloaded sample_image.png")
    except Exception as e:
        print(f"  ✗ Failed to download image: {e}")

    (TEST_LIBRARY_ROOT / ".inbox" / "test.pdf").write_text("dummy pdf content")

    print(f"✓ Test library created at: {TEST_LIBRARY_ROOT}")


def test_explore_library_tool():
    """Test ExploreLibraryTool - browse directory structure."""
    print("\n" + "=" * 60)
    print("TEST: ExploreLibraryTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [ExploreLibraryTool(manager)]
    executor = ToolExecutor(tools)

    # Test 1: Explore root
    print("\n[1] Explore root directory:")
    result = executor.execute("explore_library", {"path": "."})
    print_result(result)

    # Test 2: Explore specific subdirectory
    print("\n[2] Explore papers directory:")
    result = executor.execute("explore_library", {"path": "papers"})
    print_result(result)

    # Test 3: Non-existent path
    print("\n[3] Explore non-existent directory:")
    result = executor.execute("explore_library", {"path": "nonexistent"})
    print_result(result)


def test_move_file_tool():
    """Test MoveFileTool - move files around."""
    print("\n" + "=" * 60)
    print("TEST: MoveFileTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [MoveFileTool(manager)]
    executor = ToolExecutor(tools)

    # Setup: ensure test file exists
    test_file = TEST_LIBRARY_ROOT / ".inbox" / "test_move.pdf"
    test_file.write_text("test content")

    # Test 1: Move to existing directory
    print("\n[1] Move file to existing directory:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/test_move.pdf",
        "dest_path": "papers/test_move.pdf"
    })
    print_result(result)
    if (TEST_LIBRARY_ROOT / "papers" / "test_move.pdf").exists():
        print("  (verified: file exists at destination)")

    # Test 2: Move to new directory (should create it)
    test_file2 = TEST_LIBRARY_ROOT / ".inbox" / "test_move2.pdf"
    test_file2.write_text("test content 2")

    print("\n[2] Move file and create new directory:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/test_move2.pdf",
        "dest_path": "papers/ML/test_move2.pdf"
    })
    print_result(result)

    # Test 3: Move non-existent file
    print("\n[3] Try to move non-existent file:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/nonexistent.pdf",
        "dest_path": "papers/fail.pdf"
    })
    print_result(result)


def test_view_file_tool():
    """Test ViewFileTool - read file contents."""
    print("\n" + "=" * 60)
    print("TEST: ViewFileTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [ViewFileTool(manager)]
    executor = ToolExecutor(tools)

    # Test 1: View existing PDF file
    print("\n[1] View existing PDF file:")
    result = executor.execute("view_file", {"file_path": "papers/AI/transformer.pdf"})
    result.output["text"] = result.output["text"][:100] + "..."  # Truncate for display
    print_result(result)

    # Test 2: View existing image file
    print("\n[2] View existing image file:")
    result = executor.execute("view_file", {"file_path": "papers/AI/sample_image.png", "info_type": "content"})
    result.output["image_url"]["url"] = result.output["image_url"]["url"][:100] + "..."  # Truncate for display
    print_result(result)

    # Test 3: View non-existent file
    print("\n[3] View non-existent file:")
    result = executor.execute("view_file", {"file_path": "nonexistent.pdf"})
    print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Test tools with configurable selection")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["explore", "move", "view"],
        default=["explore", "move", "view"],
        help="Select which tools to test (default: all)",
    )
    args = parser.parse_args()

    print("Setting up test library...")
    setup_test_library()

    # Run selected tests
    if "explore" in args.tools:
        test_explore_library_tool()

    if "move" in args.tools:
        test_move_file_tool()

    if "view" in args.tools:
        test_view_file_tool()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
