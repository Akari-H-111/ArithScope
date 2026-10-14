"""Exact finite cases and numerical consistency checks. Not ML benchmark claims."""
import cmath
import itertools
import math
import unittest
from reference.arithmetic_reference import (
    descendants, descendant_prior, euler_phi, identifiable_modulus,
    mixture_responsibilities, normalized_kernel, normalized_white_features,
    phase_vector, prior_softmax, primitive_feature, ramanujan_sum,
)


class ArithmeticReferenceTests(unittest.TestCase):
    def test_c5(self):
        self.assertEqual([ramanujan_sum(5, j) for j in range(10)],
                         [4,-1,-1,-1,-1,4,-1,-1,-1,-1])

    def test_c1(self):
        self.assertTrue(all(ramanujan_sum(1,n) == 1 for n in range(-5,6)))

    def test_phi(self):
        self.assertEqual([euler_phi(n) for n in (1,2,3,5,6,12)], [1,1,2,4,2,4])

    def test_trace_kernel_matches_feature_gram(self):
        for m in range(1,18):
            for t,s in itertools.product(range(-3,4),repeat=2):
                a,b=primitive_feature(m,t),primitive_feature(m,s)
                gram=sum(v.conjugate()*w for v,w in zip(a,b))
                self.assertAlmostEqual(gram.real, normalized_kernel(m,t,s),places=12)
                self.assertAlmostEqual(gram.imag,0,places=12)

    def test_kernel_psd(self):
        for m in (1,2,4,5,9,12,15):
            points=list(range(8))
            vector=[complex(i-3,(i*i)%4-1) for i in points]
            q=sum(vector[i].conjugate()*vector[j]*normalized_kernel(m,points[i],points[j])
                  for i in range(len(points)) for j in range(len(points)))
            self.assertGreaterEqual(q.real,-1e-10)
            self.assertAlmostEqual(q.imag,0,places=10)

    def test_kernel_normalized_bounded(self):
        for m in range(1,40):
            for n in range(-20,21):
                self.assertLessEqual(abs(normalized_kernel(m,n,0)),1+1e-14)
            self.assertEqual(normalized_kernel(m,3,3),1)

    def test_identifiability_modulus(self):
        self.assertEqual(identifiable_modulus(120,[6,10]),60)
        self.assertEqual(identifiable_modulus(120,[6,10,15]),120)
        for L,H in ((120,[6,10]),(120,[6,10,15]),(210,[6,10,15]),(77,[7,14])):
            period=identifiable_modulus(L,H)
            for t,s in itertools.product(range(L),repeat=2):
                a,b=phase_vector(L,H,t),phase_vector(L,H,s)
                observed=all(abs(x-y)<1e-9 for x,y in zip(a,b))
                self.assertEqual(observed,(t-s)%period==0)

    def test_descendants_sum(self):
        for D in range(2,14):
            for m in range(1,14):
                p=descendant_prior(D,m)
                self.assertTrue(p)
                self.assertEqual(set(p),set(descendants(D,m)))
                self.assertEqual(sum(euler_phi(r) for r in p),D*euler_phi(m))
                self.assertAlmostEqual(sum(p.values()),1)

    def test_softmax_reduces_to_prior(self):
        for D,m in ((6,1),(6,5),(8,3)):
            prior=descendant_prior(D,m)
            for logit in (0.0,8.5,-20.2):
                got=prior_softmax(D,m,{r:logit for r in prior})
                for r in prior: self.assertAlmostEqual(prior[r],got[r])

    def test_invalid_router_keys(self):
        with self.assertRaises(ValueError): prior_softmax(6,1,{1:0})

    def test_white_pointwise_identity(self):
        for theta in ((0.0,0.0),(0.2,0.37),(1/3, 0.13),(-0.14,2.6)):
            v=normalized_white_features(theta,[1,2,3,5])
            self.assertAlmostEqual(sum(abs(z)**2 for z in v),1,places=13)

    def test_white_integer_collapse(self):
        v=normalized_white_features([1.0],[1,2,3])
        self.assertAlmostEqual(abs(v[0]),1)
        self.assertTrue(all(abs(z)<1e-10 for z in v[1:]))

    def test_mixture_gradient_numeric(self):
        logprior=[math.log(x) for x in (.2,.3,.5)]
        scores=[-1.2,0.8,0.1]
        p=mixture_responsibilities(logprior,scores)
        self.assertAlmostEqual(sum(p),1)
        def obj(vals):
            a=[x+y for x,y in zip(logprior,vals)]
            shift=max(a)
            return shift+math.log(sum(math.exp(x-shift) for x in a))
        for i in range(3):
            eps=1e-6
            plus=scores.copy(); minus=scores.copy()
            plus[i]+=eps; minus[i]-=eps
            self.assertAlmostEqual((obj(plus)-obj(minus))/(2*eps),p[i],places=8)


if __name__=='__main__':
    unittest.main()
