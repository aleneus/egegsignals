# egegsignals - Software for processing electrogastroenterography signals.

# Copyright (C) 2013 -- 2017 Aleksandr Popov, Aleksey Tyulpin, Anastasia Kuzmina

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from .context import egegsignals
from egegsignals import parameters as par
import unittest
import numpy as np
import generators as gen

class TestDominantFrequecy(unittest.TestCase):
    """Test suit for dominant_frequency."""
    def test_dominant_frequency_returns_value(self):
        dt = 0.5
        t, x = gen.harmonic(60, dt, 0.05)
        p = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        self.assertTrue(p != None)

    def test_dominant_frequency_1_harmonic(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        self.assertAlmostEqual(p, 0.05, places=3)

    def test_dominant_frequency_2_harmonics_in_one_section(self):
        dt = 0.5
        t, x1 = gen.harmonic(600, dt, 0.045, A=1)
        t, x2 = gen.harmonic(600, dt, 0.055, A=2)
        x = x1 + x2
        p = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        self.assertAlmostEqual(p, 0.055, places=3)
        
    def test_dominant_frequency_different_dt(self):
        dt = 0.1
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p2 = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        self.assertAlmostEqual(p1, p2, places=3)

    def test_dominant_frequency_2_harmonics_in_different_sections(self):
        dt = 0.5
        t, x1 = gen.harmonic(600, dt, 0.05, A=1)
        t, x2 = gen.harmonic(600, dt, 0.1, A=2)
        x = x1 + x2
        p = par.dominant_frequency(x, dt, par.egeg_fs['stomach'])
        self.assertAlmostEqual(p, 0.05, places=3)
        
if __name__ == '__main__':
    unittest.main()
