#!/usr/bin/env python3
"""
Bacterial DNA Sequence Analyzer using ProkBERT (Simplified)
Uses ProkBERT directly from HuggingFace without the prokbert package.
Optimized for Mac M4 laptop hardware.

Requirements:
    pip install transformers torch numpy
"""

import torch
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("ERROR: transformers package not found!")
    print("Please install it with: pip install transformers torch")
    exit(1)


class SimpleProkBERTAnalyzer:
    """
    Analyze bacterial DNA sequences using ProkBERT model.
    Simplified version that uses standard HuggingFace transformers without prokbert package.
    """
    
    def __init__(self, model_name: str = "neuralbioinfo/prokbert-mini", device: str = None):
        """
        Initialize the analyzer with ProkBERT model.
        
        Args:
            model_name: HuggingFace model identifier
                Options: 
                - "neuralbioinfo/prokbert-mini" (default)
                - "neuralbioinfo/prokbert-mini-long"
            device: 'mps', 'cuda', 'cpu', or None for auto-detection
        """
        print("="*60)
        print("Initializing ProkBERT Bacterial DNA Analyzer (Simplified)")
        print("="*60)
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Mac M1/M2/M3/M4 GPU
                print("✓ Mac GPU (MPS) detected - using Metal acceleration!")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("✓ CUDA GPU detected")
            else:
                self.device = torch.device('cpu')
                print("✓ Using CPU")
        else:
            self.device = torch.device(device)
            print(f"✓ Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"✓ Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("\nNote: ProkBERT requires trust_remote_code=True")
            print("If this fails, the model may not be accessible or you need to update transformers")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
        print("="*60 + "\n")
    
    def _kmerize_sequence(self, sequence: str, k: int = 6) -> str:
        """
        Convert DNA sequence to k-mer representation for tokenization.
        
        Args:
            sequence: DNA sequence
            k: k-mer size
            
        Returns:
            Space-separated k-mers
        """
        sequence = sequence.upper()
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            # Only add valid k-mers (A, T, G, C only)
            if all(base in 'ATGC' for base in kmer):
                kmers.append(kmer)
        return ' '.join(kmers)
    
    def get_sequence_embedding(self, sequence: str, kmer_size: int = 6) -> np.ndarray:
        """
        Get embedding vector for a bacterial DNA sequence.
        
        Args:
            sequence: DNA sequence string (A, T, G, C)
            kmer_size: Size of k-mers for tokenization (default 6)
            
        Returns:
            Numpy array of sequence embedding
        """
        # Clean and kmerize sequence
        sequence = sequence.upper().replace('N', '')
        kmerized = self._kmerize_sequence(sequence, k=kmer_size)
        
        if not kmerized:
            raise ValueError("Sequence too short or contains invalid characters")
        
        # Tokenize
        inputs = self.tokenizer(kmerized, return_tensors="pt", 
                               padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def analyze_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage."""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total = len(sequence)
        return (gc_count / total * 100) if total > 0 else 0.0
    
    def find_orfs(self, sequence: str, min_length: int = 100) -> List[Dict]:
        """
        Find Open Reading Frames (ORFs) in the sequence.
        
        Args:
            sequence: DNA sequence
            min_length: Minimum ORF length in nucleotides
            
        Returns:
            List of ORF dictionaries with start, end, length, frame
        """
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        sequence = sequence.upper()
        orfs = []
        
        # Check all 6 reading frames (3 forward, 3 reverse)
        for strand in [1, -1]:
            seq = sequence if strand == 1 else self._reverse_complement(sequence)
            
            for frame in range(3):
                i = frame
                while i < len(seq) - 2:
                    codon = seq[i:i+3]
                    
                    if codon in start_codons:
                        # Found start codon, look for stop
                        for j in range(i+3, len(seq)-2, 3):
                            stop_codon = seq[j:j+3]
                            if stop_codon in stop_codons:
                                orf_length = j - i + 3
                                if orf_length >= min_length:
                                    orfs.append({
                                        'start': i if strand == 1 else len(seq) - j - 3,
                                        'end': j + 3 if strand == 1 else len(seq) - i,
                                        'length': orf_length,
                                        'frame': frame if strand == 1 else -(frame + 1),
                                        'strand': '+' if strand == 1 else '-'
                                    })
                                break
                    i += 3
        
        return sorted(orfs, key=lambda x: x['length'], reverse=True)
    
    def _reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement.get(base, base) for base in reversed(sequence.upper()))
    
    def compare_sequences(self, seq1: str, seq2: str) -> Dict[str, float]:
        """
        Compare two bacterial sequences using embedding similarity.
        
        Args:
            seq1: First DNA sequence
            seq2: Second DNA sequence
            
        Returns:
            Dictionary with similarity metrics
        """
        print("Computing embeddings for sequence 1...")
        emb1 = self.get_sequence_embedding(seq1)
        
        print("Computing embeddings for sequence 2...")
        emb2 = self.get_sequence_embedding(seq2)
        
        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'interpretation': 'Very similar' if cosine_sim > 0.95 else 
                            'Similar' if cosine_sim > 0.85 else 
                            'Moderately similar' if cosine_sim > 0.7 else 'Different'
        }
    
    def analyze_sequence(self, sequence: str, name: str = "Unknown") -> Dict:
        """
        Comprehensive analysis of a bacterial DNA sequence.
        
        Args:
            sequence: DNA sequence string
            name: Name/identifier for the sequence
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing sequence: {name}")
        print(f"{'='*60}")
        print(f"Sequence length: {len(sequence)} bp")
        
        # Basic statistics
        gc_content = self.analyze_gc_content(sequence)
        print(f"GC content: {gc_content:.2f}%")
        
        # Find ORFs
        print("Finding Open Reading Frames...")
        orfs = self.find_orfs(sequence, min_length=100)
        print(f"Found {len(orfs)} ORFs (≥100 bp)")
        
        # Get embedding
        print("Computing sequence embedding with ProkBERT...")
        embedding = self.get_sequence_embedding(sequence)
        
        results = {
            'name': name,
            'length': len(sequence),
            'gc_content': gc_content,
            'num_orfs': len(orfs),
            'orfs': orfs[:5],  # Top 5 longest ORFs
            'embedding': embedding,
            'embedding_dim': len(embedding)
        }
        
        print("✓ Analysis complete!\n")
        return results


def main():
    """Example usage of the SimpleProkBERTAnalyzer."""
    
    print("\n" + "="*60)
    print("ProkBERT Bacterial DNA Sequence Analyzer")
    print("Optimized for Mac M4 with MPS GPU acceleration")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = SimpleProkBERTAnalyzer()
    
    # Example bacterial sequences (real E. coli genes)
    sequences = {
        "E. coli lacZ fragment": "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCCTGATGCGGTATTTTCTCCTTACGCATCTGTGCGGTATTTCACACCGCATATGGTGCACTCTCAGTACAATCTGCTCTGATGCCGCATAGTTAAGCCAGCCCCGACACCCGCCAACACCCGCTGACGCGCCCTGACGGGCTTGTCTGCTCCCGGCATCCGCTTACAGACAAGCTGTGACCGTCTCCGGGAGCTGCATGTGTCAGAGGTTTTCACCGTCATCACCGAAACGCGCGAGACGAAAGGGCCTCGTGATACGCCTATTTTTATAGGTTAATGTCATGATAATAATGGTTTCTTAGACGTCAGGTGGCACTTTTCGGGGAAATGTGCGCGGAACCCCTATTTGTTTATTTTTCTAAATACATTCAAATATGTATCCGCTCATGAGACAATAACCCTGATAAATGCTTCAATAATATTGAAAAAGGAAGAGTATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCTCACCCAGAAACGCTGGTGAAAGTAAAAGATGCTGAAGATCAGTTGGGTGCACGAGTGGGTTACATCGAACTGGATCTCAACAGCGGTAAGATCCTTGAGAGTTTTCGCCCCGAAGAACGTTTTCCAATGATGAGCACTTTTAAAGTTCTGCTATGTGGCGCGGTATTATCCCGTATTGACGCCGGGCAAGAGCAACTCGGTCGCCGCATACACTATTCTCAGAATGACTTGGTTGAGTACTCACCAGTCACAGAAAAGCATCTTACGGATGGCATGACAGTAAGAGAATTATGCAGTGCTGCCATAACCATGAGTGATAACACTGCGGCCAACTTACTTCTGACAACGATCGGAGGACCGAAGGAGCTAACCGCTTTTTTGCACAACATGGGGGATCATGTAACTCGCCTTGATCGTTGGGAACCGGAGCTGAATGAAGCCATACCAAACGACGAGCGTGACACCACGATGCCTGTAGCAATGGCAACAACGTTGCGCAAACTATTAACTGGCGAACTACTTACTCTAGCTTCCCGGCAACAATTAATAGACTGGATGGAGGCGGATAAAGTTGCAGGACCACTTCTGCGCTCGGCCCTTCCGGCTGGCTGGTTTATTGCTGATAAATCTGGAGCCGGTGAGCGTGGGTCTCGCGGTATCATTGCAGCACTGGGGCCAGATGGTAAGCCCTCCCGTATCGTAGTTATCTACACGACGGGGAGTCAGGCAACTATGGATGAACGAAATAGACAGATCGCTGAGATAGGTGCCTCACTGATTAAGCATTGGTAA",
        
        "E. coli rpoB fragment": "ATGCGTGACGACCCGCAAGCGGAACGGGCGACCACCCTGCTGCTGCTTGTCGGCATCTCCGCGCAAGGCGACGACCTGATCGAAGTGAAAGGCCTGGCGCTGCAAACCCCGGTGTTCCCCTGAACAAGACGGCGGGCATCGTCACCGGCGACGTCGTCGCGATGCTGGATGTGACCAACATCACCGACGACATCATCGACAAGCGCAACGTCACCGGCATCGTGCGCACGATCATGGGCGACGGCGACATCGACATCGACCTGACCAACGACGGCATCGTCGACGGCGTCGACGACGCGATCGACGTCGTCGACATGATCGACGGCATCGACATCAACGACGGCATGACCCTGCTGACGGTGACCGGCCTGCTGACCGGCACCGAAACCATCCTGACCGGCATCGACGGCGTCATCGACGGCATCGACATCGTCGACGGCATCGACATCGACGGCATCGACATCGACGGCATCGACATCGACGGCATCGACATCGACGGCATCGACATCGACGGCATCGAC"
    }
    
    # Analyze sequences
    results = {}
    for name, seq in sequences.items():
        results[name] = analyzer.analyze_sequence(seq, name)
    
    # Compare sequences
    print("\n" + "="*60)
    print("Sequence Comparison (ProkBERT Embeddings)")
    print("="*60)
    
    seq_names = list(sequences.keys())
    if len(seq_names) >= 2:
        comparison = analyzer.compare_sequences(
            sequences[seq_names[0]], 
            sequences[seq_names[1]]
        )
        
        print(f"\nComparing '{seq_names[0]}' vs '{seq_names[1]}':")
        print(f"  Cosine Similarity: {comparison['cosine_similarity']:.4f}")
        print(f"  Euclidean Distance: {comparison['euclidean_distance']:.4f}")
        print(f"  Interpretation: {comparison['interpretation']}")
    
    # Display top ORFs
    print("\n" + "="*60)
    print("Top Open Reading Frames")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        if result['orfs']:
            for i, orf in enumerate(result['orfs'][:3], 1):
                print(f"  ORF {i}: {orf['length']} bp, Frame: {orf['frame']}, "
                      f"Strand: {orf['strand']}, Start: {orf['start']}, End: {orf['end']}")
        else:
            print("  No ORFs found")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("ProkBERT is specifically trained on prokaryotic genomes!")
    print("="*60)


if __name__ == "__main__":
    main()
