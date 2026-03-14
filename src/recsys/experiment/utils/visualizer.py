import matplotlib.pyplot as plt


def criterion_curve(
    record: list[float],
    label: str,
    criterion: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(record, label=label)
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_curve(
    records: list[list[float]],
    labels: list[str],
    criterion: str,
    title: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    for hist, label in zip(records, labels):
        plt.plot(hist, label=label)
    plt.xlabel('EPOCH')
    plt.ylabel(criterion)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()