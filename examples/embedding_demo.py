"""Example usage of the embedding system.

Run this to see how the encoder and vector store work together.
"""

from pathlib import Path
import tempfile
from PIL import Image

from pangu_agent.library.embeddings import OpenCLIPEncoder, VectorStore


def main():
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("=" * 60)
        print("Embedding System Demo")
        print("=" * 60)

        # 1. Initialize encoder
        print("\n1. Initializing OpenCLIP encoder...")
        encoder = OpenCLIPEncoder()
        print(f"   ✓ Encoder ready: {encoder}")
        print(f"   Embedding dimension: {encoder.embedding_dim}")

        # 2. Initialize vector store
        print("\n2. Initializing vector store...")
        store = VectorStore(tmpdir / ".vector_store")
        print(f"   ✓ Store ready: {store}")

        # 3. Create sample image
        print("\n3. Creating sample image...")
        img_path = tmpdir / "cat.png"
        img = Image.new("RGB", (224, 224), color=(255, 200, 150))
        img.save(img_path)
        print(f"   ✓ Image saved to {img_path.name}")

        # 4. Generate embedding
        print("\n4. Generating embedding...")
        embedding = encoder.encode_image(img_path)
        print(f"   ✓ Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {(embedding ** 2).sum() ** 0.5:.4f}")

        # 5. Add to vector store
        print("\n5. Adding to vector store...")
        store.add(
            file_id="cat_001",
            embedding=embedding,
            metadata={"path": "animals/cat.png", "type": "image"}
        )
        print(f"   ✓ Store now contains {store.count()} embedding(s)")

        # 6. Search with text query
        print("\n6. Searching with text query: 'a cute cat'...")
        query_emb = encoder.encode_text("a cute cat")
        results = store.search(query_emb, top_k=1)

        if results:
            print(f"   ✓ Found: {results[0].file_id}")
            print(f"   Similarity: {results[0].score:.4f}")
            print(f"   Metadata: {results[0].metadata}")

        # 7. Add more embeddings
        print("\n7. Adding more embeddings...")
        text_embeddings = [
            ("Machine learning research paper", "ml_paper_001"),
            ("Deep neural networks", "nn_paper_002"),
            ("Quantum computing introduction", "quantum_001"),
        ]

        for text, file_id in text_embeddings:
            emb = encoder.encode_text(text)
            store.add(
                file_id=file_id,
                embedding=emb,
                metadata={"description": text}
            )

        print(f"   ✓ Store now contains {store.count()} embedding(s)")

        # 8. Search for similar content
        print("\n8. Searching for 'neural network papers'...")
        query_emb = encoder.encode_text("neural network papers")
        results = store.search(query_emb, top_k=3)

        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.file_id} (similarity: {result.score:.4f})")
            print(f"      {result.metadata.get('description', 'N/A')}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
