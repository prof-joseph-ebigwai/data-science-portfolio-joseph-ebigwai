"""
E-News Express – A/B Testing and Statistical Analysis
Portfolio Code Extract

Author: Joseph Kayefor Ebigwai

This cleaned extract is based on the statistical-testing workflow developed
in the E-News Express project. It demonstrates an independent t-test,
two-proportion z-test, Chi-square test of independence, and one-way ANOVA.

The original course dataset is intentionally not included.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.proportion import proportions_ztest


def compare_time_spent(df):
    """Test whether users spend more time on the new landing page."""
    new_page = df.loc[
        df["landing_page"] == "new",
        "time_spent_on_the_page",
    ]
    old_page = df.loc[
        df["landing_page"] == "old",
        "time_spent_on_the_page",
    ]

    t_stat, two_tailed_p = stats.ttest_ind(
        new_page,
        old_page,
        equal_var=True,
    )

    # One-tailed alternative: mean time on new page > mean time on old page.
    one_tailed_p = two_tailed_p / 2

    return {
        "t_statistic": t_stat,
        "p_value": one_tailed_p,
        "new_page_mean": new_page.mean(),
        "old_page_mean": old_page.mean(),
    }


def compare_conversion_rates():
    """Compare conversion rates for new and old landing pages."""
    conversions = np.array([33, 21])
    visitors = np.array([50, 50])

    z_stat, p_value = proportions_ztest(
        conversions,
        visitors,
        alternative="larger",
    )

    return {
        "z_statistic": z_stat,
        "p_value": p_value,
        "new_page_conversion_rate": conversions[0] / visitors[0],
        "old_page_conversion_rate": conversions[1] / visitors[1],
    }


def test_conversion_language_independence(df):
    """Test whether conversion is associated with preferred language."""
    contingency_table = pd.crosstab(
        df["language_preferred"],
        df["converted"],
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected,
    }


def compare_time_by_language_on_new_page(df):
    """Test whether time spent differs across language groups on the new page."""
    new_page_df = df[df["landing_page"] == "new"]

    english_time = new_page_df.loc[
        new_page_df["language_preferred"] == "English",
        "time_spent_on_the_page",
    ]
    french_time = new_page_df.loc[
        new_page_df["language_preferred"] == "French",
        "time_spent_on_the_page",
    ]
    spanish_time = new_page_df.loc[
        new_page_df["language_preferred"] == "Spanish",
        "time_spent_on_the_page",
    ]

    f_stat, p_value = f_oneway(
        english_time,
        french_time,
        spanish_time,
    )

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
    }


# Example usage:
#
# time_result = compare_time_spent(df)
# conversion_result = compare_conversion_rates()
# language_conversion_result = test_conversion_language_independence(df)
# language_time_result = compare_time_by_language_on_new_page(df)
#
# print(time_result)
# print(conversion_result)
# print(language_conversion_result)
# print(language_time_result)
