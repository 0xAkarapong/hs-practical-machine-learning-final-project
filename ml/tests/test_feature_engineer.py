import pandas as pd

from src._02_feature_engineer import add_interaction_features


def test_add_interaction_features_adds_one_column_per_section():
    df = pd.DataFrame(
        {
            "name_word_count": [3, 5, 0],
            "section_vitamins": [1, 0, 1],
            "section_herbs": [0, 1, 0],
            "shop_region_northern": [1, 0, 0],
        }
    )
    out = add_interaction_features(df)

    assert out.shape[0] == df.shape[0]  # row count unchanged
    assert "x_section_vitamins_wordcount" in out.columns
    assert "x_section_herbs_wordcount" in out.columns
    # interaction is section dummy * word count; shop_region_* is left alone
    assert out["x_section_vitamins_wordcount"].tolist() == [3, 0, 0]
    assert out["x_section_herbs_wordcount"].tolist() == [0, 5, 0]
    assert "name_word_count" in out.columns  # originals preserved
