"""methods for tracing and evaluating ff integration."""
import sparrowpy.form_factor.integration as intg
import sparrowpy.testing.exact_ff_solutions as ffsol
import sparrowpy.geometry as geo
import numpy as np

class integration_test:
    """Class for convenient testing of integration methods."""

    def __init__(self, gtype="a", ww=1, hh=1, ll=1):
        self._select_geometry(gtype,ww,hh,ll)
        self.int_method = intg.surface_ff_naive
        self.result=None
        self.error=None

    def _select_geometry(self,gtype="a",ww=1,hh=1,ll=1):
        """Select geometry (parallel or coincident line patches)."""

        match gtype:
            case "a":
                self.patch_1 = geo.Polygon(
                    points=[[0, 0, 0],
                            [ww, 0, 0],
                            [ww, 0, hh],
                            [0, 0, hh]],
                    normal=[0, 1, 0],
                    up_vector=[1, 0, 0],
                )

                self.patch_2 = geo.Polygon(
                    points=[
                        [0, ll, 0],
                        [0, ll, hh],
                        [ww, ll, hh],
                        [ww, ll, 0]],
                    normal=[0, -1, 0],
                    up_vector=[1, 0, 0],
                )

                self.solution = ffsol.parallel_patches(ww, hh, ll)

            case "b":
                self.patch_1 = geo.Polygon(
                    points=[
                        [0, 0, 0],
                        [0, ll, 0],
                        [0, ll, hh],
                        [0, 0, hh]],
                    normal=[1, 0, 0],
                    up_vector=[1, 0, 0],
                )

                self.patch_2 = geo.Polygon(
                    points=[[0, 0, 0],
                            [ww, 0, 0],
                            [ww, ll, 0],
                            [0, ll, 0]],
                    normal=[0, 0, 1],
                    up_vector=[1, 0, 0],
                )

                self.solution = ffsol.perpendicular_patch_coincidentline(
                                ww, hh, ll,
                            )
            case _:
                ValueError(
                    f"{gtype} does not correspond to any known geometries.",
                    )

    def integrate(self):
        """Run integration method."""
        self.result = self.int_method(self.patch_1.pts,
                               self.patch_2.pts,
                               self.patch_1.normal,
                               self.patch_2.normal,
                               self.patch_1.area,
                               self.patch_2.area)

    def calc_error(self):
        """Calculate integration error."""
        if self.result is None:
            self.integrate()

        self.error=self.result/self.solution
        self.error_absolute=self.solution-self.result

    def geometry(self):
        """Return geometry data."""
        return(self.patch_1,self.patch_2)

    def print_results(self):
        """Print errors."""
        if self.error is None:
            self.calc_error()

        print(f'\nrelative ff error:       {self.error-1:.4%}')
        print(f'gain per 100 bounces:      {10*np.log10(self.error**100):.1f}'+
               'dB\n')

    def _update_kwargs(self):
        self.kwargs_list=[]
        for method in self.int_method:
            temp={}
            match method:
                case intg.surface_ff:
                    temp["patch_i"]=self.patch_1.pts
                    temp["patch_j"]=self.patch_2.pts
                    temp["patch_i_normal"]=self.patch_1.normal
                    temp["patch_j_normal"]=self.patch_2.normal

if __name__=="__main__":
    test=integration_test()
    test.print_results()
