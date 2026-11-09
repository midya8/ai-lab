Mean over 3 seed(s), 1800 training rows, 600 clean test rows. AUROC / P@k are for the CV-margin ranking from the named model; the two baselines are model-independent (in-sample softmax loss, k-NN label disagreement).

| data | model | noise | true -> est. rate | AUROC margin | AUROC in-sample | AUROC kNN-disagree | P@k | F1 both | F1 noise-rate | test acc: noisy -> pruned (oracle) |
|---|---|---|---|---|---|---|---|---|---|---|
| blobs (overlap) | softmax | symmetric 10% | 0.10 -> 0.13 | 0.981 | 0.984 | 0.969 | 0.811 | 0.812 | 0.786 | 0.885 -> 0.892 (0.895) |
| blobs (overlap) | softmax | symmetric 20% | 0.20 -> 0.24 | 0.984 | 0.986 | 0.969 | 0.881 | 0.861 | 0.852 | 0.879 -> 0.888 (0.895) |
| blobs (overlap) | softmax | symmetric 40% | 0.40 -> 0.45 | 0.975 | 0.976 | 0.939 | 0.906 | 0.884 | 0.887 | 0.866 -> 0.881 (0.895) |
| blobs (overlap) | softmax | pairwise 30% | 0.30 -> 0.39 | 0.863 | 0.846 | 0.866 | 0.673 | 0.681 | 0.708 | 0.724 -> 0.824 (0.895) |
| blobs (overlap) | softmax | asymmetric 30% | 0.30 -> 0.37 | 0.919 | 0.907 | 0.903 | 0.773 | 0.790 | 0.796 | 0.807 -> 0.853 (0.895) |
| blobs (overlap) | kNN | symmetric 10% | 0.10 -> 0.15 | 0.972 | 0.984 | 0.969 | 0.785 | 0.783 | 0.751 | 0.885 -> 0.892 (0.895) |
| blobs (overlap) | kNN | symmetric 20% | 0.20 -> 0.26 | 0.974 | 0.986 | 0.969 | 0.840 | 0.843 | 0.837 | 0.879 -> 0.882 (0.895) |
| blobs (overlap) | kNN | symmetric 40% | 0.40 -> 0.47 | 0.957 | 0.976 | 0.939 | 0.872 | 0.860 | 0.861 | 0.866 -> 0.880 (0.895) |
| blobs (overlap) | kNN | pairwise 30% | 0.30 -> 0.37 | 0.918 | 0.846 | 0.866 | 0.765 | 0.764 | 0.807 | 0.724 -> 0.871 (0.895) |
| blobs (overlap) | kNN | asymmetric 30% | 0.30 -> 0.37 | 0.938 | 0.907 | 0.903 | 0.793 | 0.797 | 0.816 | 0.807 -> 0.875 (0.895) |
| two-mode classes | softmax | symmetric 10% | 0.10 -> 0.23 | 0.929 | 0.939 | 0.985 | 0.646 | 0.586 | 0.571 | 0.771 -> 0.777 (0.777) |
| two-mode classes | softmax | symmetric 20% | 0.20 -> 0.33 | 0.933 | 0.937 | 0.977 | 0.746 | 0.714 | 0.711 | 0.772 -> 0.773 (0.777) |
| two-mode classes | softmax | symmetric 40% | 0.40 -> 0.51 | 0.921 | 0.925 | 0.940 | 0.811 | 0.792 | 0.788 | 0.756 -> 0.771 (0.777) |
| two-mode classes | softmax | pairwise 30% | 0.30 -> 0.45 | 0.817 | 0.799 | 0.880 | 0.626 | 0.623 | 0.641 | 0.675 -> 0.726 (0.777) |
| two-mode classes | softmax | asymmetric 30% | 0.30 -> 0.45 | 0.898 | 0.886 | 0.926 | 0.722 | 0.733 | 0.744 | 0.729 -> 0.766 (0.777) |
| two-mode classes | kNN | symmetric 10% | 0.10 -> 0.13 | 0.988 | 0.939 | 0.985 | 0.833 | 0.819 | 0.803 | 0.771 -> 0.779 (0.777) |
| two-mode classes | kNN | symmetric 20% | 0.20 -> 0.23 | 0.983 | 0.937 | 0.977 | 0.870 | 0.858 | 0.861 | 0.772 -> 0.778 (0.777) |
| two-mode classes | kNN | symmetric 40% | 0.40 -> 0.45 | 0.964 | 0.925 | 0.940 | 0.888 | 0.866 | 0.866 | 0.756 -> 0.777 (0.777) |
| two-mode classes | kNN | pairwise 30% | 0.30 -> 0.35 | 0.934 | 0.799 | 0.880 | 0.790 | 0.787 | 0.824 | 0.675 -> 0.772 (0.777) |
| two-mode classes | kNN | asymmetric 30% | 0.30 -> 0.34 | 0.960 | 0.886 | 0.926 | 0.841 | 0.835 | 0.862 | 0.729 -> 0.773 (0.777) |
