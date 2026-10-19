import { resolve } from "path";

import { pluginReact } from "@rsbuild/plugin-react";
import { defineConfig } from "@rstest/core";

export default defineConfig({
  plugins: [pluginReact()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  output: {
    // Streamdown imports KaTeX CSS as a side effect. Bundle these packages so
    // Rsbuild processes that CSS import instead of Node trying to load it.
    bundleDependencies: ["streamdown", "katex"],
  },
  // Component render tests cover the visual fixes that pure projection tests
  // cannot see (see tests/unit/components/workspace/nir).
  include: ["tests/unit/**/*.test.ts", "tests/unit/**/*.test.tsx"],
});
