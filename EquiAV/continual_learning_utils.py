import numpy as np
import matplotlib.pyplot as plt
import random

def store_audio_decision(
    obs_counts: np.ndarray,
    stored_counts: np.ndarray,
    incoming_labels: list[int],
    stored_examples: list,
    replace_prob: float
):
    """
    Determine whether to store a new multi-label example and, if storing,
    which stored example should be evicted.

    This method handles multi-label data where each sample may contain
    multiple class labels. Class distributions are computed over label
    occurrences.

    Parameters:
        obs_counts : np.ndarray
            Array of shape [K]. Each entry obs_counts[k] is the number of times
            class k has been observed in the incoming data stream. These counts
            represent label occurrences.

        stored_counts : np.ndarray
            Array of shape [K]. stored_counts[k] is the number of times class k
            appears in the current reservoir.

        incoming_labels : list[int]
            The labels for the incoming example.

        stored_examples: list
            A list representing the stored reservoir. Each element should be a
            dict or object containing at least:
                sample["labels"] = list of class indices for that example.

            Example:
                samples[i]["labels"] = [0, 2]

    replace_prob : float
        A global scalar controlling how often replacement should occur.
        The final store probability is:
            mean(P_obs[label] for label in incoming_labels) * replace_prob

    Returns:
        should_store : bool
            Whether to store the incoming sample.

        class_to_reduce : int or None
            If storing, this is the class selected for reduction (chosen from
            overrepresented classes based on proportional differential).
            None if no class is selected (e.g., fallback eviction).

        sample_to_evict : int or None
            Index into `samples` of the example chosen for eviction.
            None if should_store is False or if reservoir is empty.
    """
    # Safety check
    if obs_counts.sum() == 0:
        raise ValueError("Observed counts sum to zero; cannot compute distribution.")

    # If reservoir empty, store immediately
    if stored_counts.sum() == 0:
        return True, None, None

    # Compute probability distributions
    P_obs = obs_counts / obs_counts.sum()
    P_stored = stored_counts / stored_counts.sum()

    # Compute probability of storing the incoming example
    incoming_label_probs = np.array([P_obs[c] for c in incoming_labels])
    P_store = incoming_label_probs.mean() * replace_prob

    should_store = np.random.rand() < P_store
    if not should_store:
        return False, None, None

    # Class differential for eviction
    differential = P_stored - P_obs
    positive_mask = differential > 0

    stored_total_examples = len(stored_examples)

    # If no overrepresented classes, then evict random example
    if not positive_mask.any():
        sample_to_evict = np.random.choice(stored_total_examples)
        return True, None, sample_to_evict

    # Weighted random choice among overrepresented classes
    eviction_probs = differential.copy()
    eviction_probs[~positive_mask] = 0
    eviction_probs /= eviction_probs.sum()

    class_to_reduce = np.random.choice(len(eviction_probs), p=eviction_probs)

    # Select a sample containing that class
    candidate_idxs = [
        i for i, s in enumerate(stored_examples)
        if class_to_reduce in s["labels"]
    ]

    if len(candidate_idxs) == 0:
        sample_to_evict = np.random.choice(stored_total_examples)
    else:
        sample_to_evict = np.random.choice(candidate_idxs)

    return True, class_to_reduce, sample_to_evict


def simulate():
    np.random.seed(0)
    random.seed(0)

    K = 5
    REPLACE_PROB = 0.9
    RESERVOIR_SIZE = 50
    STEPS = 100

    # Initial observed distribution
    obs_counts = np.random.randint(50, 200, size=K)

    # Initial stored reservoir
    stored_examples = []
    stored_counts = np.zeros(K, dtype=int)

    for _ in range(RESERVOIR_SIZE):
        num_labels = np.random.randint(1, 3)
        labels = list(np.random.choice(K, size=num_labels, replace=False))
        stored_examples.append({"labels": labels})
        for c in labels:
            stored_counts[c] += 1

    # Tracking
    stored_history = []
    obs_history = []
    total_evictions = 0

    # Simulate incoming data
    for step in range(STEPS):

        # Generate an incoming sample
        num_labels = np.random.randint(1, 3)
        incoming_labels = list(np.random.choice(K, size=num_labels, replace=False))

        # UPDATE OBSERVED COUNTS (important!)
        for c in incoming_labels:
            obs_counts[c] += 1

        # Decide whether to store / which to evict
        should_store, class_to_reduce, sample_to_evict = store_audio_decision(
            obs_counts,
            stored_counts,
            incoming_labels,
            stored_examples,
            REPLACE_PROB
        )

        if should_store:

            # Evict if needed
            if sample_to_evict is not None:
                old_labels = stored_examples[sample_to_evict]["labels"]
                for c in old_labels:
                    stored_counts[c] -= 1
                stored_examples[sample_to_evict] = {"labels": incoming_labels}
                total_evictions += 1
            else:
                stored_examples.append({"labels": incoming_labels})

            # Add new counts
            for c in incoming_labels:
                stored_counts[c] += 1

        # Track proportions
        stored_history.append(stored_counts / stored_counts.sum())
        obs_history.append(obs_counts / obs_counts.sum())

    stored_history = np.array(stored_history)
    obs_history = np.array(obs_history)

    # Plot the results
    plt.figure(figsize=(12, 7))

    # Use a fixed color map (tab10 is good for up to ~10 classes)
    colors = plt.cm.tab10(np.arange(K))

    for k in range(K):
        plt.plot(
            stored_history[:, k],
            label=f"Stored class {k}",
            color=colors[k],
            linestyle='-',
            linewidth=2
        )

        plt.plot(
            obs_history[:, k],
            label=f"Observed class {k} (true)",
            color=colors[k],
            linestyle='--',
            alpha=0.7,
            linewidth=2
        )

    
    print(f"Total evictions during simulation: {total_evictions}")

    plt.title("Stored vs Observed Distribution Over Time")
    plt.xlabel("Time (incoming examples)")
    plt.ylabel("Proportion")
    plt.legend(ncol=2, fontsize=8)

    # Add a label inside the plot
    plt.text(
        x=1.0,
        y=1.05,
        s=f"Replacement Prob = {REPLACE_PROB}", 
        transform=plt.gca().transAxes, 
        fontsize=10, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5)
    )

    plt.grid(True)
    plt.show()



# Run the simulation
if __name__ == "__main__":
    simulate()
