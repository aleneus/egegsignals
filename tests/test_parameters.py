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

class TestEnergy(unittest.TestCase):
    """Test suit for energy."""
    def test_energy_unit(self):
        dt = 0.5
        x = np.array([1,1,1,1,1,1,1,1,1,1]) # ten units # 5 sec
        p = par.energy(x, dt, [0, 1])
        self.assertEqual(p, 5)
    
    def test_energy_rectangle(self):
        dt = 0.5
        x = np.array([2,2,2,2,2,2,2,2,2,2]) # 5 sec
        p = par.energy(x, dt, [0, 1])
        self.assertEqual(p, 20)
        
    def test_energy_dont_rely_on_dt(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.energy(x, dt, par.egeg_fs['stomach'])
        dt = 0.05
        t, x = gen.harmonic(600, dt, 0.05)
        p2 = par.energy(x, dt, par.egeg_fs['stomach'])
        self.assertLess(abs(p1/p2 - 1), 0.01)

    def test_energy_proportional_to_length(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.energy(x, dt, par.egeg_fs['stomach'])
        t, x = gen.harmonic(600*10, dt, 0.05)
        p2 = par.energy(x, dt, par.egeg_fs['stomach'])
        self.assertLess(abs(p2/p1/10 - 1), 0.01)
        
class TestPower(unittest.TestCase):
    """Test suit for energy."""
    def test_power_unit(self):
        dt=1
        x = np.array([1,1,1,1,1,1,1,1,1,1]) # 10 sec
        p = par.power(x, dt, (0, 1))
        self.assertEqual(p, 1)
    
    def test_power_rectangle(self):
        dt=0.5
        x = np.array([2,2,2,2,2,2,2,2,2,2]) # 5 sec
        p = par.power(x, dt, (0, 1))
        self.assertEqual(p, 4)
        
    def test_power_dont_rely_on_length(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.power(x, dt, par.egeg_fs['stomach'])
        t, x = gen.harmonic(600*10, dt, 0.05)
        p2 = par.power(x, dt, par.egeg_fs['stomach'])
        self.assertLess(abs(p2/p1 - 1), 0.01)

    def test_power_dont_rely_on_dt(self):
        dt = 0.05
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.power(x, dt, par.egeg_fs['stomach'])
        dt = 0.01
        t, x = gen.harmonic(600, dt, 0.05)
        p2 = par.power(x, dt, par.egeg_fs['stomach'])
        self.assertLess(abs(p2/p1 - 1), 0.01)
        
class TestRhythmicity(unittest.TestCase):
    """Test suit for energy."""
    def test_rhythmicity_rely_on_power(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.rhythmicity(x, dt, par.egeg_fs['stomach'])
        t, x = gen.harmonic(600, dt, 0.05, A = 2)
        p2 = par.rhythmicity(x, dt, par.egeg_fs['stomach'])
        self.assertLess(p1, p2)
        
class TestRhythmicityNorm(unittest.TestCase):
    """Test suit for energy."""
    def test_rhythmicity_rely_on_power(self):
        dt = 0.5
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.rhythmicity_norm(x, dt, par.egeg_fs['stomach'])
        t, x = gen.harmonic(600, dt, 0.05, A = 2)
        p2 = par.rhythmicity_norm(x, dt, par.egeg_fs['stomach'])
        self.assertEqual(p1/p2, 1)
        
    def test_rhythmicity_norm_dont_rely_on_dt(self):
        dt = 0.05
        t, x = gen.harmonic(600, dt, 0.05)
        p1 = par.rhythmicity_norm(x, dt, par.egeg_fs['stomach'])
        dt = 0.01
        t, x = gen.harmonic(600, dt, 0.05, A = 2)
        p2 = par.rhythmicity_norm(x, dt, par.egeg_fs['stomach'])
        self.assertLess(abs(p1/p2-1), 0.01)

class TestSTFT(unittest.TestCase):
    def test_stft_number_of_spectrums(self):
        dt = 0.5
        t, x = gen.harmonic(650, dt, 0.05)
        Xs = par.stft(x, dt, window_len = 120)
        self.assertEqual(len(Xs), 10)
        
    def test_stft_short_signal(self):
        dt = 1
        t, x = gen.harmonic(60, dt, 0.05)
        Xs = par.stft(x, dt, window_len = 120)
        self.assertEqual(len(Xs), 0)

if __name__ == '__main__':
    unittest.main()
