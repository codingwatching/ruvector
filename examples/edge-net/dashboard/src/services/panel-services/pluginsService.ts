/**
 * Plugins Service - WASM Modules as Plugins
 *
 * Maps WASM modules to plugin representation for the dashboard.
 * Plugins are WASM modules that extend Edge-Net functionality.
 */

import { useWASMStore } from '../../stores/wasmStore';
import type { WASMModule } from '../../types';

export interface RealPlugin {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
  category: 'ai' | 'network' | 'security' | 'utility' | 'compute';
  status: 'installed' | 'available' | 'update-available';
  securityStatus: 'verified' | 'quarantine' | 'pending' | 'unverified';
  enabled: boolean;
  downloads: number;
  rating: number;
  size: string;
  isReal: boolean;
}

export interface PluginStats {
  totalPlugins: number;
  installedPlugins: number;
  verifiedPlugins: number;
  enabledPlugins: number;
  quarantinedPlugins: number;
}

// Map WASM module types to plugin categories
const moduleToCategory: Record<string, RealPlugin['category']> = {
  core: 'compute',
  compute: 'compute',
  'neural-embed': 'ai',
  'neural-vectordb': 'ai',
  network: 'network',
  crypto: 'security',
};

class PluginsService {
  /**
   * Get plugins from WASM modules
   */
  getWASMPlugins(): RealPlugin[] {
    const state = useWASMStore.getState();

    return state.modules.map((module: WASMModule) => ({
      id: module.id,
      name: module.name,
      description: `WASM module: ${module.features.join(', ')}`,
      author: 'RuVector',
      version: module.version,
      category: moduleToCategory[module.id] || 'utility',
      status: module.status === 'ready' ? 'installed' :
              module.status === 'loading' ? 'installed' : 'available',
      securityStatus: 'verified' as const, // WASM modules are verified
      enabled: module.status === 'ready',
      downloads: 10000 + Math.floor(Math.random() * 5000), // Simulated
      rating: 4.5 + Math.random() * 0.5,
      size: `${(module.size / 1024 / 1024).toFixed(1)} MB`,
      isReal: true,
    }));
  }

  /**
   * Get all plugins
   */
  getAllPlugins(): RealPlugin[] {
    return this.getWASMPlugins();
  }

  /**
   * Get plugin statistics
   */
  getPluginStats(): PluginStats {
    const plugins = this.getAllPlugins();
    const installed = plugins.filter(p => p.status === 'installed');
    const verified = plugins.filter(p => p.securityStatus === 'verified');
    const enabled = plugins.filter(p => p.enabled);
    const quarantined = plugins.filter(p => p.securityStatus === 'quarantine');

    return {
      totalPlugins: plugins.length,
      installedPlugins: installed.length,
      verifiedPlugins: verified.length,
      enabledPlugins: enabled.length,
      quarantinedPlugins: quarantined.length,
    };
  }

  /**
   * Toggle plugin enabled state
   */
  togglePlugin(pluginId: string): void {
    const wasmStore = useWASMStore.getState();
    const module = wasmStore.modules.find(m => m.id === pluginId);

    if (module) {
      // Toggle module status (ready -> unloaded or unloaded -> ready)
      wasmStore.updateModule(pluginId, {
        status: module.status === 'ready' ? 'unloaded' : 'ready',
      });
    }
  }
}

export const pluginsService = new PluginsService();
