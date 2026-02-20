"""methods for tracing and evaluating ff integration."""
import sparrowpy.form_factor.integration as intg
import sparrowpy.testing.exact_ff_solutions as ffsol
import sparrowpy.geometry as geo
import numpy as np

class integration_test:
    """Class for convenient testing of integration methods."""

    def __init__(self, gtype="a", ww=1, hh=1, ll=1):
        self._select_geometry(gtype,ww,hh,ll)
        self.set_int_method()
        self.result=None
        self.error=None
        self.set_nsamples(9)

    def set_nsamples(self,nsamples:int):
        self.nsamples = nsamples
        self.order = int(nsamples**.5)-1

    def set_poly_order(self,order:int):
        self.order = order
        self.nsamples = (order+1)**2

    def set_int_method(self,op1="contour",op2="poly"):

        self.intg_op1=op1
        if op1.lower()=="contour":
            self.intg_method=intg.contour_ff
            self.intg_op2=op2
        elif op1.lower()=="nusselt":
            self.intg_method=intg.surface_ff_Nusselt
            self.intg_op2=op2
        elif op1.lower()=="naive":
            self.intg_method=intg.surface_ff_naive
            self.intg_op2=op2


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

        if self.intg_method == intg.surface_ff_naive:
            self.result = self.intg_method(self.patch_1.pts,
                                self.patch_2.pts,
                                self.patch_1.normal,
                                self.patch_2.normal,
                                self.patch_1.area,
                                self.patch_2.area,
                                self.nsamples,
                                self.intg_op2)
        elif self.intg_method == intg.surface_ff_Nusselt:
            self.result = self.intg_method(self.patch_1.pts,
                                self.patch_2.pts,
                                self.patch_1.normal,
                                self.patch_2.normal,
                                self.patch_1.area,
                                self.nsamples,
                                self.intg_op2)
        elif self.intg_method == intg.contour_ff:
            self.result = self.intg_method(self.patch_1.pts,
                                self.patch_2.pts,
                                self.patch_1.area,
                                self.intg_op2,
                                self.order)


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

        print(f'\n -> relative ff error:       {self.error-1:.4%}')
        print(f' -> gain per 100 bounces:      {10*np.log10(self.error**100):.1f}'+
               'dB\n')

    def print_stats(self):
        print('\n\n############################################')
        print(f'Form Factor approach: {self.intg_op1}')
        print(f'Internal integral approach: {self.intg_op2}')
        if (self.intg_op1.lower()=='contour' or
            self.intg_op2.lower()=='poly'):
            print(f'Newton-Cotes polynomial order: {self.order}')
        else:
            print(f'# of samples in each patch: {self.nsamples}')


if __name__=="__main__":
    test=integration_test()
    test.print_stats()
    test.print_results()
