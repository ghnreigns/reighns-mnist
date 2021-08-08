def forward_test(x, model):
    y = model(x)
    print("[forward test]")
    print("input:\t{}\noutput:\t{}".format(x.shape, y.shape))
    print("output value", y)
