"""
Usage examples (archived to reduce top-level noise).
"""

def example_training():
    import train
    train.main()


def example_single_prediction():
    from ..predict import predict_win_probability
    # Placeholder example: user can edit with real battle
    battle = {}
    return predict_win_probability(battle)


if __name__ == '__main__':
    print('Archived examples module. Import specific examples as needed.')
