from src.data.AmazonReviewTokenized import AmazonReviewTokenized, ToTensor


def test_constructor():
    dataset = AmazonReviewTokenized(transform=ToTensor())
    assert type(dataset) == AmazonReviewTokenized
    assert type(dataset.transform) == ToTensor


def test_len():
    dataset = AmazonReviewTokenized()

    # Can't check actual length as it depends on the size of the processed
    # data file
    assert type(len(dataset)) == int


def test_getitem():
    dataset = AmazonReviewTokenized(transform=ToTensor())

    item = next(iter(dataset))
    assert item is not None
