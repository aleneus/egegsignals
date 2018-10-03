# egegsignals - Software for processing electrogastroenterography signals.

# Copyright (C) 2013 -- 2018 Aleksandr Popov

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

"""Tests for parameters."""

import sys
import os
import unittest
import random
import numpy as np
from scipy.fftpack import fft

sys.path.insert(0, os.path.abspath('.'))
from egegsignals import parameters as par


def harmonic(length, sampling_period, freq, amp=1):
    """Generate harmonis signal."""
    tdata = np.arange(0, length, sampling_period)
    xdata = amp * np.cos(2 * np.pi * freq * tdata)
    return xdata


class TestDominantFrequecy(unittest.TestCase):
    """Test suit for dominant_frequency."""
    def test_df_returns_value(self):
        """Dominant frequency just calculated."""
        sampling_period = 0.5
        xdata = harmonic(60, sampling_period, 0.05)
        val = par.dominant_frequency(abs(fft(xdata)),
                                     sampling_period, par.egeg_fs['stomach'])
        self.assertTrue(val is not None)

    def test_df_harmonic(self):
        """Dominant frequency of harmonic signal."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val = par.dominant_frequency(abs(fft(xdata)),
                                     sampling_period, par.egeg_fs['stomach'])
        self.assertAlmostEqual(val, 0.05, places=3)

    def test_df_2_harmonics_in_section(self):
        """Dominant frequency of signal of two harmonic having
        different amplitudes but similar frequencies."""
        sampling_period = 0.5
        xdata1 = harmonic(600, sampling_period, 0.045, amp=1)
        xdata2 = harmonic(600, sampling_period, 0.055, amp=2)
        xdata = xdata1 + xdata2
        val = par.dominant_frequency(abs(fft(xdata)),
                                     sampling_period, par.egeg_fs['stomach'])
        self.assertAlmostEqual(val, 0.055, places=3)

    def test_dominant_frequency_dt(self):
        """DF does not rely on sampling period."""
        sampling_period = 0.1
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.dominant_frequency(abs(fft(xdata)),
                                      sampling_period, par.egeg_fs['stomach'])
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val2 = par.dominant_frequency(abs(fft(xdata)),
                                      sampling_period, par.egeg_fs['stomach'])
        self.assertAlmostEqual(val1, val2, places=3)

    def test_df_2_harmonics(self):
        """Dominant frequency of signal of two harmonic having
        different amplitudes."""
        sampling_period = 0.5
        xdata1 = harmonic(600, sampling_period, 0.05, amp=1)
        xdata2 = harmonic(600, sampling_period, 0.1, amp=2)
        xdata = xdata1 + xdata2
        val = par.dominant_frequency(abs(fft(xdata)),
                                     sampling_period, par.egeg_fs['stomach'])
        self.assertAlmostEqual(val, 0.05, places=3)


class TestEnergy(unittest.TestCase):
    """Test suit for energy."""
    def test_energy_unit(self):
        """Energy of unit rectangle signal."""
        sampling_period = 0.5
        xdata = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # ten units, 5 sec
        val = par.energy(abs(fft(xdata)), sampling_period, [0, 1])
        self.assertEqual(val, 5)

    def test_energy_rectangle(self):
        """Energy of rectangle signal."""
        sampling_period = 0.5
        xdata = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # 5 sec
        val = par.energy(abs(fft(xdata)), sampling_period, [0, 1])
        self.assertEqual(val, 20)

    def test_energy_dt(self):
        """Energy does not rely on sampling period."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.energy(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        sampling_period = 0.05
        xdata = harmonic(600, sampling_period, 0.05)
        val2 = par.energy(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        self.assertLess(abs(val1/val2 - 1), 0.01)

    def test_energy_length(self):
        """Energy proportional to the length of signal."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.energy(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        xdata = harmonic(600*10, sampling_period, 0.05)
        val2 = par.energy(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        self.assertLess(abs(val2/val1/10 - 1), 0.01)


class TestPower(unittest.TestCase):
    """Test suit for energy."""
    def test_power_unit(self):
        """Power calculated correctly for rectangular unit signal."""
        sampling_period = 1
        xdata = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 10 sec
        val = par.power(abs(fft(xdata)), sampling_period, (0, 1))
        self.assertEqual(val, 1)

    def test_power_rectangle(self):
        """Power calculated correctly for rectangular signal."""
        sampling_period = 0.5
        xdata = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # 5 sec
        value = par.power(abs(fft(xdata)), sampling_period, (0, 1))
        self.assertEqual(value, 4)

    def test_power_dont_rely_on_length(self):
        """Power does not rely on the length of signal."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.power(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        xdata = harmonic(600*10, sampling_period, 0.05)
        val2 = par.power(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        self.assertLess(abs(val2/val1 - 1), 0.01)

    def test_power_dont_rely_on_dt(self):
        """Power does not rely on sampling period"""
        sampling_period = 0.05
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.power(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        sampling_period = 0.01
        xdata = harmonic(600, sampling_period, 0.05)
        val2 = par.power(abs(fft(xdata)), sampling_period, par.egeg_fs['stomach'])
        self.assertLess(abs(val2/val1 - 1), 0.01)


class TestRhythmicity(unittest.TestCase):
    """Test suit for energy."""
    def test_rhythmicity_rely_on_power(self):
        """Rhythmicity coefficient (GS) rely on power."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.rhythmicity(abs(fft(xdata)),
                               sampling_period, par.egeg_fs['stomach'])
        xdata = harmonic(600, sampling_period, 0.05, amp=2)
        val2 = par.rhythmicity(abs(fft(xdata)),
                               sampling_period, par.egeg_fs['stomach'])
        self.assertLess(val1, val2)


class TestRhythmicityNorm(unittest.TestCase):
    """Test suit for energy."""
    def test_rhythmicity_norm_power(self):
        """Normalized rhythmicity coefficient does not rely on
        power."""
        sampling_period = 0.5
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.rhythmicity_norm(abs(fft(xdata)),
                                    sampling_period, par.egeg_fs['stomach'])
        xdata = harmonic(600, sampling_period, 0.05, amp=2)
        val2 = par.rhythmicity_norm(abs(fft(xdata)),
                                    sampling_period, par.egeg_fs['stomach'])
        self.assertEqual(val1, val2)

    def test_rhythmicity_norm_dt(self):
        """Normalized rhythmicity coefficient does not rely on
        sampling period."""
        sampling_period = 0.05
        xdata = harmonic(600, sampling_period, 0.05)
        val1 = par.rhythmicity_norm(abs(fft(xdata)),
                                    sampling_period, par.egeg_fs['stomach'])
        sampling_period = 0.01
        xdata = harmonic(600, sampling_period, 0.05, amp=2)
        val2 = par.rhythmicity_norm(abs(fft(xdata)),
                                    sampling_period, par.egeg_fs['stomach'])
        self.assertLess(abs(val1/val2 - 1), 0.01)


class TestDFIC(unittest.TestCase):
    """Tests for dominant frequency instability coefficient."""
    def test_dfic_long_harmonic(self):
        """DFIC of harmonic with zero average is about zero."""
        sampling_period = 0.5
        xdata = harmonic(60*40, sampling_period, 0.05)
        value = par.dfic(par.egeg_fs['stomach'], xdata, sampling_period,
                         nseg=1200, nstep=120)
        self.assertAlmostEqual(value, 0, places=3)

    def test_dfic_random_signal(self):
        """DFIC of random signal with zero average is very small."""
        sampling_period = 0.25
        xdata = np.array([random.randint(-1000, 1000) for i in range(4800)])
        value = par.dfic((0, 2), xdata, sampling_period,
                         nseg=1200, nstep=120)
        self.assertLess(0.3, value)


class TestNextOrgan(unittest.TestCase):
    """Tests for getting of next organ name."""
    def test_next_for_stomach(self):
        """Duodenum goes after the stomach."""
        next_organ_name = par.next_organ_name('stomach')
        self.assertEqual(next_organ_name, 'duodenum')

    def test_next_for_colon(self):
        """There is no organ after the colon."""
        next_organ_name = par.next_organ_name('colon')
        self.assertEqual(next_organ_name, None)


if __name__ == '__main__':
    unittest.main()
