{pkgs}: {
  deps = [
    pkgs.libGLU
    pkgs.libGL
    pkgs.xvfb-run
    pkgs.scrot
  ];
}
