# Executed reference test receipt (2026-10-14)

This record covers **only this handoff package's standard-library arithmetic reference**, not any PyTorch implementation or prior research repository verification.

- Command: `python3 -m unittest discover -s tests -v`
- Interpreter: `3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]`
- Platform: `Linux-6.18.44-x86_64-with-glibc2.41`
- Exit code: `0`
- Result: **PASS**

```text
test_c1 (test_arithmetic_reference.ArithmeticReferenceTests.test_c1) ... ok
test_c5 (test_arithmetic_reference.ArithmeticReferenceTests.test_c5) ... ok
test_descendants_sum (test_arithmetic_reference.ArithmeticReferenceTests.test_descendants_sum) ... ok
test_identifiability_modulus (test_arithmetic_reference.ArithmeticReferenceTests.test_identifiability_modulus) ... ok
test_invalid_router_keys (test_arithmetic_reference.ArithmeticReferenceTests.test_invalid_router_keys) ... ok
test_kernel_normalized_bounded (test_arithmetic_reference.ArithmeticReferenceTests.test_kernel_normalized_bounded) ... ok
test_kernel_psd (test_arithmetic_reference.ArithmeticReferenceTests.test_kernel_psd) ... ok
test_mixture_gradient_numeric (test_arithmetic_reference.ArithmeticReferenceTests.test_mixture_gradient_numeric) ... ok
test_phi (test_arithmetic_reference.ArithmeticReferenceTests.test_phi) ... ok
test_softmax_reduces_to_prior (test_arithmetic_reference.ArithmeticReferenceTests.test_softmax_reduces_to_prior) ... ok
test_trace_kernel_matches_feature_gram (test_arithmetic_reference.ArithmeticReferenceTests.test_trace_kernel_matches_feature_gram) ... ok
test_white_integer_collapse (test_arithmetic_reference.ArithmeticReferenceTests.test_white_integer_collapse) ... ok
test_white_pointwise_identity (test_arithmetic_reference.ArithmeticReferenceTests.test_white_pointwise_identity) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.317s

OK
```
