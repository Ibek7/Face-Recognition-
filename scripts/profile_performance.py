#!/usr/bin/env python3
"""
Performance profiling script for Face Recognition System.
Profiles different components and generates detailed reports.
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path
from io import StringIO
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_detection(iterations=100):
    """Profile face detection performance."""
    print(f"\nProfiling face detection ({iterations} iterations)...")
    
    import numpy as np
    from src.detection import FaceDetector
    
    detector = FaceDetector()
    
    # Generate random test image
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    def run_detection():
        for _ in range(iterations):
            detector.detect_faces(test_image)
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_detection()
    profiler.disable()
    
    return profiler


def profile_embedding_generation(iterations=50):
    """Profile embedding generation performance."""
    print(f"\nProfiling embedding generation ({iterations} iterations)...")
    
    import numpy as np
    from src.embeddings import FaceEmbeddingManager
    
    manager = FaceEmbeddingManager()
    
    # Generate random face crop
    face_crop = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    
    def run_embedding():
        for _ in range(iterations):
            manager.generate_embeddings_from_image(face_crop)
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_embedding()
    profiler.disable()
    
    return profiler


def profile_database_operations(iterations=100):
    """Profile database operations."""
    print(f"\nProfiling database operations ({iterations} iterations)...")
    
    from src.database import DatabaseManager
    import numpy as np
    
    db = DatabaseManager("sqlite:///:memory:")
    
    # Add some test data
    for i in range(10):
        person = db.add_person(f"Person_{i}", f"Test person {i}")
        embedding = np.random.randn(512).astype(np.float32)
        db.add_face_embedding(embedding, person.id, f"test_{i}.jpg", 0.9, "simple")
    
    def run_queries():
        for _ in range(iterations):
            db.list_persons()
            query_emb = np.random.randn(512).astype(np.float32)
            db.search_similar_faces(query_emb, threshold=0.5, top_k=5)
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_queries()
    profiler.disable()
    
    return profiler


def print_profile_stats(profiler, top_n=20):
    """Print profiling statistics."""
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    print(s.getvalue())


def save_profile_report(profiler, output_file):
    """Save detailed profile report to file."""
    with open(output_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()
    
    print(f"  Detailed report saved to: {output_file}")


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description="Profile Face Recognition System")
    parser.add_argument("--component", choices=["detection", "embedding", "database", "all"],
                       default="all", help="Component to profile")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for profiling")
    parser.add_argument("--output-dir", type=str, default="profiling_results",
                       help="Output directory for detailed reports")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top functions to display")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Face Recognition System - Performance Profiling")
    print("=" * 70)
    
    components = []
    if args.component == "all":
        components = ["detection", "embedding", "database"]
    else:
        components = [args.component]
    
    for component in components:
        print(f"\n{'=' * 70}")
        print(f"Profiling: {component.upper()}")
        print(f"{'=' * 70}")
        
        start_time = time.time()
        
        if component == "detection":
            profiler = profile_detection(args.iterations)
        elif component == "embedding":
            profiler = profile_embedding_generation(args.iterations // 2)
        elif component == "database":
            profiler = profile_database_operations(args.iterations)
        
        elapsed = time.time() - start_time
        
        print(f"\nTotal Time: {elapsed:.2f}s")
        print(f"\nTop {args.top_n} Functions by Cumulative Time:")
        print("-" * 70)
        print_profile_stats(profiler, args.top_n)
        
        # Save detailed report
        report_file = output_dir / f"{component}_profile.txt"
        save_profile_report(profiler, report_file)
    
    print(f"\n{'=' * 70}")
    print("Profiling Complete")
    print(f"{'=' * 70}")
    print(f"\nDetailed reports saved in: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review the profiling results above")
    print("  2. Check detailed reports for bottlenecks")
    print("  3. Optimize slow functions")
    print("  4. Re-run profiling to verify improvements")
    print()


if __name__ == "__main__":
    main()
