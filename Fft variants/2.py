#https://stackoverflow.com/a/45512765/3922292

def makeSpectrum(E, dx, dy, upsample=10):
    """
    Convert a time-domain array `E` to the frequency domain via 2D FFT. `dx` and
    `dy` are sample spacing in x (left-right, 1st axis) and y (up-down, 0th
    axis) directions. An optional `upsample > 1` will zero-pad `E` to obtain an
    upsampled spectrum.

    Returns `(spectrum, xf, yf)` where `spectrum` contains the 2D FFT of `E`. If
    `Ny, Nx = spectrum.shape`, `xf` and `yf` will be vectors of length `Nx` and
    `Ny` respectively, containing the frequencies corresponding to each pixel of
    `spectrum`.

    The returned spectrum is zero-centered (via `fftshift`). The 2D FFT, and
    this function, assume your input `E` has its origin at the top-left of the
    array. If this is not the case, i.e., your input `E`'s origin is translated
    away from the first pixel, the returned `spectrum`'s phase will *not* match
    what you expect, since a translation in the time domain is a modulation of
    the frequency domain. (If you don't care about the spectrum's phase, i.e.,
    only magnitude, then you can ignore all these origin issues.)
    """
    zeropadded = np.array(E.shape) * upsample
    F = fft.fftshift(fft.fft2(E, zeropadded)) / E.size
    xf = fft.fftshift(fft.fftfreq(zeropadded[1], d=dx))
    yf = fft.fftshift(fft.fftfreq(zeropadded[0], d=dy))
    return (F, xf, yf)


def extents(f):
    "Convert a vector into the 2-element extents vector imshow needs"
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


def plotSpectrum(F, xf, yf):
    "Plot a spectrum array and vectors of x and y frequency spacings"
    plt.figure()
    plt.imshow(abs(F),
               aspect="equal",
               interpolation="none",
               origin="lower",
               extent=extents(xf) + extents(yf))
    plt.colorbar()
    plt.xlabel('f_x (Hz)')
    plt.ylabel('f_y (Hz)')
    plt.title('|Spectrum|')
    plt.show()


if __name__ == '__main__':
    # In seconds
    x = np.linspace(0, 4, 20)
    y = np.linspace(0, 4, 30)
    # Uncomment the next two lines and notice that the spectral peak is no
    # longer equal to 1.0! That's because `makeSpectrum` expects its input's
    # origin to be at the top-left pixel, which isn't the case for the following
    # two lines.
    # x = np.linspace(.123 + 0, .123 + 4, 20)
    # y = np.linspace(.123 + 0, .123 + 4, 30)

    # Sinusoid frequency, in Hz
    x0 = 1.9
    y0 = -2.9

    # Generate data
    im = np.exp(2j * np.pi * (y[:, np.newaxis] * y0 + x[np.newaxis, :] * x0))

    # Generate spectrum and plot
    spectrum, xf, yf = makeSpectrum(im, x[1] - x[0], y[1] - y[0])
    plotSpectrum(spectrum, xf, yf)

    # Report peak
    peak = spectrum[:, np.isclose(xf, x0)][np.isclose(yf, y0)]
    peak = peak[0, 0]
    print('spectral peak={}'.format(peak))
