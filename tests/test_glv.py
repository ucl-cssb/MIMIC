import unittest

import numpy
from mimicsim import glv_sim


class TestGlv(unittest.TestCase):
    def test_glv_1_species_random(self):
        glv = glv_sim.GlvSim(num_species=1)
        glv.print()
        self.assertEqual(glv.nsp, 1)
        self.assertEqual(len(glv.mu), 1)
        self.assertEqual(glv.M.shape, (1, 1))

    def test_glv_1_species(self):
        glv = glv_sim.GlvSim(num_species=1, mu=[1], M=numpy.array([[-1]]))
        glv.print()
        self.assertEqual(glv.nsp, 1)
        self.assertEqual(len(glv.mu), 1)
        self.assertEqual(glv.M.shape, (1, 1))
        self.assertEqual(glv.mu[0], 1)
        self.assertEqual(glv.M[0][0], -1)

    def test_glv_1_stationary_species_sim(self):
        glv = glv_sim.GlvSim(num_species=1, mu=[0], M=numpy.array([[0]]))
        glv.print()
        self.assertEqual(glv.nsp, 1)
        self.assertEqual(len(glv.mu), 1)
        self.assertEqual(glv.M.shape, (1, 1))
        self.assertEqual(glv.mu[0], 0)
        self.assertEqual(glv.M[0][0], 0)

        times = numpy.linspace(0, 10, 11)
        init_species = [1]
        s_obs, init_species, mu, M = glv.simulate(times, init_species)
        self.assertEqual(len(s_obs), 11)
        self.assertEqual(len(s_obs[0]), 1)
        self.assertEqual(len(init_species), 1)
        self.assertEqual(len(mu), 1)
        self.assertEqual(len(M), 1)
        self.assertEqual(len(M[0]), 1)
        self.assertEqual(init_species[0], 1)
        self.assertEqual(mu[0], 0)
        self.assertEqual(M[0][0], 0)
        for s in s_obs:
            self.assertEqual(s[0], 1)

        glv = glv_sim.GlvSim(num_species=1, mu=[1], M=numpy.array([[-1]]))
        glv.print()
        times = numpy.linspace(0, 10, 11)
        init_species = [1]
        s_obs, init_species, mu, M = glv.simulate(times, init_species)
        for s in s_obs:
            self.assertEqual(s[0], 1)

    def test_glv_1_exponential_species_sim(self):
        glv = glv_sim.GlvSim(num_species=1, mu=[1], M=numpy.array([[0]]))
        glv.print()
        self.assertEqual(glv.nsp, 1)
        self.assertEqual(len(glv.mu), 1)
        self.assertEqual(glv.M.shape, (1, 1))
        self.assertEqual(glv.mu[0], 1)
        self.assertEqual(glv.M[0][0], 0)

        times = numpy.linspace(0, 10, 11)
        init_species = [1]
        s_obs, init_species, mu, M = glv.simulate(times, init_species)
        self.assertEqual(len(s_obs), 11)
        self.assertEqual(len(s_obs[0]), 1)
        self.assertEqual(len(init_species), 1)
        self.assertEqual(len(mu), 1)
        self.assertEqual(len(M), 1)
        self.assertEqual(len(M[0]), 1)
        self.assertEqual(init_species[0], 1)
        self.assertEqual(mu[0], 1)
        self.assertEqual(M[0][0], 0)
        for i, s in enumerate(s_obs):
            self.assertAlmostEqual(s[0], numpy.exp(i), places=2)

    def test_glv_2_species_random(self):
        glv = glv_sim.GlvSim(num_species=2)
        glv.print()
        self.assertEqual(glv.nsp, 2)
        self.assertEqual(len(glv.mu), 2)
        self.assertEqual(glv.M.shape, (2, 2))

    def test_glv_2_species(self):
        glv = glv_sim.GlvSim(num_species=2, mu=[1, 2], M=numpy.array([[-1, 0.5], [0, -2]]))
        glv.print()
        self.assertEqual(glv.nsp, 2)
        self.assertEqual(len(glv.mu), 2)
        self.assertEqual(glv.M.shape, (2, 2))
        self.assertEqual(glv.mu[0], 1)
        self.assertEqual(glv.mu[1], 2)
        self.assertEqual(glv.M[0][0], -1)
        self.assertEqual(glv.M[0][1], 0.5)
        self.assertEqual(glv.M[1][0], 0)
        self.assertEqual(glv.M[1][1], -2)

    def test_glv_2_no_interaction_species_sim(self):
        glv = glv_sim.GlvSim(num_species=2, mu=[1, 2], M=numpy.array([[-1, 0], [0, -1]]))
        glv.print()
        times = numpy.linspace(0, 100, 11)
        init_species = [1, 1]
        s_obs, init_species, mu, M = glv.simulate(times, init_species)
        self.assertEqual(s_obs.shape, (11, 2))
        self.assertAlmostEqual(s_obs[10][0], 1)
        self.assertAlmostEqual(s_obs[10][1], 2)


if __name__ == '__main__':
    unittest.main()
