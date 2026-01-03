/**
 * usePlugins Hook - Real-time plugin data from WASM modules
 */

import { useState, useEffect, useCallback } from 'react';
import { pluginsService, type RealPlugin, type PluginStats } from '../services/panel-services/pluginsService';
import { useWASMStore } from '../stores/wasmStore';

export function usePlugins() {
  const [plugins, setPlugins] = useState<RealPlugin[]>([]);
  const [stats, setStats] = useState<PluginStats>({
    totalPlugins: 0,
    installedPlugins: 0,
    verifiedPlugins: 0,
    enabledPlugins: 0,
    quarantinedPlugins: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  // Subscribe to WASM store changes
  const wasmModules = useWASMStore(state => state.modules);

  // Refresh plugin data
  const refresh = useCallback(() => {
    const allPlugins = pluginsService.getAllPlugins();
    const pluginStats = pluginsService.getPluginStats();
    setPlugins(allPlugins);
    setStats(pluginStats);
    setIsLoading(false);
  }, []);

  // Toggle plugin
  const togglePlugin = useCallback((pluginId: string) => {
    pluginsService.togglePlugin(pluginId);
    refresh();
  }, [refresh]);

  // Initial load and periodic refresh
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh, wasmModules]);

  return {
    plugins,
    stats,
    isLoading,
    refresh,
    togglePlugin,
  };
}
