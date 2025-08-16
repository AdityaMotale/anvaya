{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells = {
          python = pkgs.mkShell {
            name = "dev-python";
            buildInputs = with pkgs; [
              gcc
              pkg-config
              python314
              ruff
              uv
              pyright
            ];

            shellHook = ''
              echo " : $(python3 --version)"
            '';
          };
        };
      }
    );
}
