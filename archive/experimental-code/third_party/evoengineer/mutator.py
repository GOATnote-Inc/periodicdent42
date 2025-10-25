"""
EvoEngineer Mutator: Parameter mutation strategies
"""
import random
from typing import List, Optional
from .optimizer import Candidate, SearchSpace


class ParameterMutator:
    """Mutates kernel parameters to generate new candidates"""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
    
    def mutate(self, parent: Candidate, mutation_rate: float = 0.3) -> Candidate:
        """Generate a mutated child from parent"""
        
        # Start with parent's parameters
        params = {
            "BLOCK_M": parent.BLOCK_M,
            "BLOCK_N": parent.BLOCK_N,
            "BLOCK_K": parent.BLOCK_K,
            "NUM_WARPS": parent.NUM_WARPS,
            "STAGES": parent.STAGES,
            "UNROLL": parent.UNROLL,
            "CP_ASYNC": parent.CP_ASYNC,
            "SWIZZLE": parent.SWIZZLE,
            "HALF2": parent.HALF2,
        }
        
        # Mutate each parameter with mutation_rate probability
        for param_name, param_value in params.items():
            if random.random() < mutation_rate:
                options = getattr(self.search_space, param_name)
                # Pick a different value if possible
                new_options = [x for x in options if x != param_value]
                if new_options:
                    params[param_name] = random.choice(new_options)
        
        # Create new candidate
        child = Candidate(
            BLOCK_M=params["BLOCK_M"],
            BLOCK_N=params["BLOCK_N"],
            BLOCK_K=params["BLOCK_K"],
            NUM_WARPS=params["NUM_WARPS"],
            STAGES=params["STAGES"],
            UNROLL=params["UNROLL"],
            CP_ASYNC=params["CP_ASYNC"],
            SWIZZLE=params["SWIZZLE"],
            HALF2=params["HALF2"],
            generation=parent.generation + 1,
            parent_hash=parent.hash,
        )
        
        return child
    
    def crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Generate child by combining two parents"""
        
        # Randomly select parameters from each parent
        child = Candidate(
            BLOCK_M=random.choice([parent1.BLOCK_M, parent2.BLOCK_M]),
            BLOCK_N=random.choice([parent1.BLOCK_N, parent2.BLOCK_N]),
            BLOCK_K=random.choice([parent1.BLOCK_K, parent2.BLOCK_K]),
            NUM_WARPS=random.choice([parent1.NUM_WARPS, parent2.NUM_WARPS]),
            STAGES=random.choice([parent1.STAGES, parent2.STAGES]),
            UNROLL=random.choice([parent1.UNROLL, parent2.UNROLL]),
            CP_ASYNC=random.choice([parent1.CP_ASYNC, parent2.CP_ASYNC]),
            SWIZZLE=random.choice([parent1.SWIZZLE, parent2.SWIZZLE]),
            HALF2=random.choice([parent1.HALF2, parent2.HALF2]),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_hash=f"{parent1.hash}+{parent2.hash}",
        )
        
        return child
    
    def local_search(self, candidate: Candidate) -> List[Candidate]:
        """Generate neighbors by changing one parameter at a time"""
        neighbors = []
        
        # Try adjacent tile sizes
        for attr in ["BLOCK_M", "BLOCK_N", "BLOCK_K"]:
            options = getattr(self.search_space, attr)
            current = getattr(candidate, attr)
            idx = options.index(current) if current in options else 0
            
            # Try next larger
            if idx + 1 < len(options):
                neighbor = self._clone_with_change(candidate, attr, options[idx + 1])
                if neighbor.estimate_smem_bytes() <= 49152:
                    neighbors.append(neighbor)
            
            # Try next smaller
            if idx - 1 >= 0:
                neighbor = self._clone_with_change(candidate, attr, options[idx - 1])
                neighbors.append(neighbor)
        
        return neighbors
    
    def _clone_with_change(self, candidate: Candidate, attr: str, new_value) -> Candidate:
        """Clone candidate with one parameter changed"""
        params = {
            "BLOCK_M": candidate.BLOCK_M,
            "BLOCK_N": candidate.BLOCK_N,
            "BLOCK_K": candidate.BLOCK_K,
            "NUM_WARPS": candidate.NUM_WARPS,
            "STAGES": candidate.STAGES,
            "UNROLL": candidate.UNROLL,
            "CP_ASYNC": candidate.CP_ASYNC,
            "SWIZZLE": candidate.SWIZZLE,
            "HALF2": candidate.HALF2,
        }
        params[attr] = new_value
        
        return Candidate(
            **params,
            generation=candidate.generation + 1,
            parent_hash=candidate.hash,
        )
