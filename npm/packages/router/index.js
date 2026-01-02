const { platform, arch } = process;
const path = require('path');

// Platform mapping for @ruvector/router
const platformMap = {
  'linux': {
    'x64': { package: '@ruvector/router-linux-x64-gnu', file: 'ruvector-router.linux-x64-gnu.node' },
    'arm64': { package: '@ruvector/router-linux-arm64-gnu', file: 'ruvector-router.linux-arm64-gnu.node' }
  },
  'darwin': {
    'x64': { package: '@ruvector/router-darwin-x64', file: 'ruvector-router.darwin-x64.node' },
    'arm64': { package: '@ruvector/router-darwin-arm64', file: 'ruvector-router.darwin-arm64.node' }
  },
  'win32': {
    'x64': { package: '@ruvector/router-win32-x64-msvc', file: 'ruvector-router.win32-x64-msvc.node' }
  }
};

function loadNativeModule() {
  const platformInfo = platformMap[platform]?.[arch];

  if (!platformInfo) {
    // Return null instead of throwing - allows SemanticRouter fallback
    return null;
  }

  // Try local .node file first (for development and bundled packages)
  try {
    const localPath = path.join(__dirname, platformInfo.file);
    return require(localPath);
  } catch (localError) {
    // Fall back to platform-specific package
    try {
      return require(platformInfo.package);
    } catch (error) {
      // Return null to allow fallback mode
      return null;
    }
  }
}

const SemanticRouter = require('./semantic-router');
const nativeModule = loadNativeModule();

module.exports = {
  // Native exports (may be null if not available)
  VectorDb: nativeModule?.VectorDb,
  DistanceMetric: nativeModule?.DistanceMetric ?? {
    Euclidean: 0,
    Cosine: 1,
    DotProduct: 2,
    Manhattan: 3
  },
  // High-level SemanticRouter (always available)
  SemanticRouter,
};
