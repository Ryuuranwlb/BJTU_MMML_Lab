"""Test SearchFilesService - run directly to see service execution trajectory."""

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.services.search_files import SearchFilesService
from pangu_agent.tools import (
    ExploreLibraryTool,
    ViewFileTool,
    SearchLibraryTool,
    FinishTool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_LIBRARY_ROOT = Path(__file__).parent.parent / "runs" / "test_search_files_library"


def setup_test_library():
    """Create a test library with some sample files."""
    # Clean up existing directory
    if TEST_LIBRARY_ROOT.exists():
        shutil.rmtree(TEST_LIBRARY_ROOT)
        print(f"✓ Removed existing library: {TEST_LIBRARY_ROOT.name}")

    # Create test library structure
    TEST_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    (TEST_LIBRARY_ROOT / ".inbox").mkdir(exist_ok=True)

    # Create some directories
    nlp_dir = TEST_LIBRARY_ROOT / "nlp" / "transformers"
    cv_dir = TEST_LIBRARY_ROOT / "computer_vision" / "cnn"
    nlp_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created test library at: {TEST_LIBRARY_ROOT}")

    # Download sample papers
    print("Downloading sample papers...")

    papers = [
        {
            "url": "https://arxiv.org/pdf/1706.03762.pdf",
            "path": nlp_dir / "attention_is_all_you_need.pdf",
            "name": "Attention Is All You Need",
        },
        {
            "url": "https://arxiv.org/pdf/1810.04805.pdf",
            "path": nlp_dir / "bert.pdf",
            "name": "BERT",
        },
        {
            "url": "https://arxiv.org/pdf/1409.1556.pdf",
            "path": cv_dir / "vgg.pdf",
            "name": "VGG",
        },
    ]

    for paper in papers:
        try:
            print(f"  Downloading {paper['name']}...")
            urlretrieve(paper["url"], paper["path"])
            print(f"  ✓ Downloaded to {paper['path'].relative_to(TEST_LIBRARY_ROOT)}")
        except Exception as e:
            print(f"  ✗ Failed to download {paper['name']}: {e}")
            raise RuntimeError(f"Failed to download test paper: {e}")

    # Download a sample image
    img_url = "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"
    img_path = cv_dir / "pytorch_logo.png"
    try:
        print(f"  Downloading PyTorch logo...")
        urlretrieve(img_url, img_path)
        print(f"  ✓ Downloaded to {img_path.relative_to(TEST_LIBRARY_ROOT)}")
    except Exception as e:
        print(f"  ✗ Failed to download image: {e}")
        raise RuntimeError(f"Failed to download test image: {e}")

    print("✓ Test library setup complete")


def test_search_transformers(manager: LibraryManager):
    """Test searching for transformer-related files."""
    print("\n" + "=" * 60)
    print("TEST: Search for Transformer Papers")
    print("=" * 60)

    llm_client = LLMClient(log=False)
    tools = [
        SearchLibraryTool(manager),
        ViewFileTool(manager),
        ExploreLibraryTool(manager),
        FinishTool(),
    ]

    service = SearchFilesService(manager, llm_client, tools)

    query = "papers about attention mechanism and transformers in NLP"
    print(f"\n[Query]: {query}")

    result = service.search(query)

    print(f"\n[Result]:")
    print(f"  Success: {result['success']}")
    print(f"  Files found: {len(result.get('files', []))}")
    print(f"  Iterations: {result.get('iterations', 'N/A')}")

    if result["success"]:
        print(f"\n[Files]:")
        for i, file in enumerate(result["files"], 1):
            print(f"  {i}. {file}")

        print(f"\n[Observation]:")
        print(f"  {result['observation']}")
    else:
        print(f"\n[Error]: {result.get('error', 'Unknown error')}")

    print("\n✓ Transformer search test completed!")


def test_search_cnn(manager: LibraryManager):
    """Test searching for CNN-related files."""
    print("\n" + "=" * 60)
    print("TEST: Search for CNN Papers")
    print("=" * 60)

    llm_client = LLMClient(log=False)
    tools = [
        SearchLibraryTool(manager),
        ViewFileTool(manager),
        ExploreLibraryTool(manager),
        FinishTool(),
    ]

    service = SearchFilesService(manager, llm_client, tools)

    query = "convolutional neural networks for image classification"
    print(f"\n[Query]: {query}")

    result = service.search(query)

    print(f"\n[Result]:")
    print(f"  Success: {result['success']}")
    print(f"  Files found: {len(result.get('files', []))}")
    print(f"  Iterations: {result.get('iterations', 'N/A')}")

    if result["success"]:
        print(f"\n[Files]:")
        for i, file in enumerate(result["files"], 1):
            print(f"  {i}. {file}")

        print(f"\n[Observation]:")
        print(f"  {result['observation']}")
    else:
        print(f"\n[Error]: {result.get('error', 'Unknown error')}")

    print("\n✓ CNN search test completed!")


def test_search_all_files(manager: LibraryManager):
    """Test searching for all files in the library."""
    print("\n" + "=" * 60)
    print("TEST: Search for All Deep Learning Papers")
    print("=" * 60)

    llm_client = LLMClient(log=False)
    tools = [
        SearchLibraryTool(manager),
        ViewFileTool(manager),
        ExploreLibraryTool(manager),
        FinishTool(),
    ]

    service = SearchFilesService(manager, llm_client, tools)

    query = "all papers related to deep learning and neural networks"
    print(f"\n[Query]: {query}")

    result = service.search(query)

    print(f"\n[Result]:")
    print(f"  Success: {result['success']}")
    print(f"  Files found: {len(result.get('files', []))}")
    print(f"  Iterations: {result.get('iterations', 'N/A')}")

    if result["success"]:
        print(f"\n[Files]:")
        for i, file in enumerate(result["files"], 1):
            print(f"  {i}. {file}")

        print(f"\n[Observation]:")
        print(f"  {result['observation']}")
    else:
        print(f"\n[Error]: {result.get('error', 'Unknown error')}")

    print("\n✓ All files search test completed!")


def main():
    parser = argparse.ArgumentParser(description="Test SearchFilesService with configurable selection")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["transformers", "cnn", "all_files", "all"],
        default=["all"],
        help="Select which tests to run (default: all)",
    )
    args = parser.parse_args()

    print("Setting up test library...")
    setup_test_library()

    # Determine which tests to run
    run_all = "all" in args.tests
    tests_to_run = args.tests if not run_all else ["transformers", "cnn", "all_files"]

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    print("✓ Library manager initialized")

    # Run selected tests
    if "transformers" in tests_to_run:
        test_search_transformers(manager)

    if "cnn" in tests_to_run:
        test_search_cnn(manager)

    if "all_files" in tests_to_run:
        test_search_all_files(manager)

    print("\n" + "=" * 60)
    print("All SearchFilesService tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
