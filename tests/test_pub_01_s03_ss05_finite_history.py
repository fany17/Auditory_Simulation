import unittest
import torch
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder,VARIANTS,pcc_loss


class FiniteHistoryTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(1024);torch.set_num_threads(2)

    def test_local_has_zero_past_dependence(self):
        model=FiniteHistoryDecoder('local',width=8).eval();x=torch.randn(1,40,192,64);changed=x.clone();changed[:,:20]+=9
        with torch.no_grad():a=model(x);b=model(changed)
        self.assertTrue(torch.equal(a[:,0],b[:,0]));self.assertTrue(torch.equal(a,b))

    def test_all_shapes_gradients(self):
        x=torch.randn(1,40,192,64);target=torch.randn(1,20,192,1)
        for variant in VARIANTS:
            model=FiniteHistoryDecoder(variant,width=8,gru_hidden=24);prediction=model(x)
            self.assertEqual(tuple(prediction.shape),(1,20,192,1));loss=pcc_loss(prediction,target);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_same_present_counterfactual_and_no_future(self):
        model=FiniteHistoryDecoder('full',width=8).eval();x=torch.randn(1,40,192,64);changed=x.clone();changed[:,21:]+=8
        with torch.no_grad():a=model(x);b=model(changed)
        self.assertTrue(torch.equal(a[:,0],b[:,0]))
        changed=x.clone();changed[:,:20]+=8
        with torch.no_grad():c=model(changed)
        self.assertGreater(float((a[:,0]-c[:,0]).abs().max()),1e-8)

    def test_query_batch_equals_separate_contiguous_context(self):
        model=FiniteHistoryDecoder('full',width=8).eval();x=torch.randn(1,40,192,64)
        with torch.no_grad():
            all_out=model(x);local,tokens=model.encode(x);readout=model.base.memory(tokens[:,:21])[:,-1]
            single=model.queries(local[:,20:21],tokens[:,20:21],tokens[:,:20][:,None])
        self.assertTrue(torch.allclose(single[:,0],all_out[:,0],atol=1e-6))
        self.assertEqual(tuple(readout.shape),(1,8))

    def test_new_records_never_inherit_state(self):
        model=FiniteHistoryDecoder('gru',width=8,gru_hidden=24).eval();a=torch.randn(1,40,192,64);b=torch.randn_like(a)
        with torch.no_grad():before=model(b);model(a);after=model(b)
        self.assertTrue(torch.equal(before,after))
