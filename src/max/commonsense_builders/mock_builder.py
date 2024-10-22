
def build_commonsense(expectation):
    return dict(
        prior=dict(foo=1,
            xIntent=['[mock xIntent]'],
            xNeed=['[mock xNeed]'],
            xAttr=['mock xAttr']),
        posterior=dict(
            xReact=['[mock xReact]'],
            xWant=['[mock xWant]'],
            xEffect=['[mock xEffect]'])
    )
