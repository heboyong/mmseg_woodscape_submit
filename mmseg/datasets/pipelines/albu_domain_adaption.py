import os
import albumentations as A
from ..builder import PIPELINES


@PIPELINES.register_module()
class AlbuDomainAdaption(object):
    """Apply Albu domain adaption methods

    """

    def __init__(self,
                 domain_adaption_type: str = 'ALL',
                 target_dir: str = None,
                 p: float = 0.5) -> None:
        self.domain_adaption_type = domain_adaption_type
        self.target_dir = target_dir
        self.p = p
        self.target_list = [os.path.join(self.target_dir, target_image) for target_image in os.listdir(self.target_dir)]
        assert self.domain_adaption_type in ["HistogramMatching", "FDA", "PixelDistributionAdaptation", 'ALL']
        assert len(os.listdir(target_dir)) > 0

    def __call__(self, results: dict) -> dict:

        if self.domain_adaption_type == "HistogramMatching":
            aug = A.Compose([A.HistogramMatching(self.target_list, p=self.p)])
        elif self.domain_adaption_type == "FDA":
            aug = A.Compose([A.FDA(self.target_list, p=self.p)])
        elif self.domain_adaption_type == "PixelDistributionAdaptation":
            aug = A.Compose([A.PixelDistributionAdaptation(self.target_list, p=self.p)])
        else:
            aug = A.Compose(
                [A.OneOf(
                    [A.PixelDistributionAdaptation(self.target_list, p=1.0),
                     A.FDA(self.target_list, p=1.0),
                     A.PixelDistributionAdaptation(self.target_list, p=1.0)])], p=self.p)

        adaption_result = aug(image=results['img'])['image']
        results['img'] = adaption_result

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'domain adaption type={self.domain_adaption_type}, '
        repr_str += f'target dir={self.target_dir}, '
        repr_str += f'p={self.p})'
        return repr_str
