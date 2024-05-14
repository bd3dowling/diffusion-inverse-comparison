# tests

Where tests live. We use [pytest](https://docs.pytest.org/en/8.2.x/). Tests at present just check
basic functionality of dashboard functions. They require manually downloading the model checkpoint
file before running, and as such aren't included in CI. In future, better mocking to circumvent
this should be added.
