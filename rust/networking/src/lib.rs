use std::{env, str::FromStr, sync::Arc};

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;

pub mod discovery;
pub mod swarm;

pub fn is_valid_zid(identity: &str) -> bool {
    let mut iter = identity.chars();
    iter.next()
        .is_some_and(|c| ('1'..='9').contains(&c) || ('a'..='f').contains(&c))
        && iter.all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
        && identity.len() <= 32
}

pub fn cfg(identity: &str, listen_port: u16) -> Result<zenoh::Config> {
    assert!(is_valid_zid(identity));
    assert!(identity.len() <= 32);
    assert!(listen_port != 0, "must used defined listen port");
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    if let Ok(endpoints) = env::var("EXO_CONNECT_ENDPOINTS") {
        let endpoints = endpoints
            .split(",")
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(|endpoint| format!("\"{endpoint}\""))
            .collect::<Vec<_>>();
        if !endpoints.is_empty() {
            log::warn!(
                "EXO_CONNECT_ENDPOINTS forcing Zenoh connect/endpoints={}",
                format!("[{}]", endpoints.join(","))
            );
            cfg.insert_json5("connect/endpoints", &format!("[{}]", endpoints.join(",")))?;
        }
    }
    cfg.insert_json5("adminspace/enabled", "true")?;
    //cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    //cfg.insert_json5("timestamping/enabled", "true")?;
    cfg.insert_json5("plugins/storage_manager/__required__", "true")?;
    cfg.insert_json5(
        "plugins/storage_manager/storages/mem1",
        r#"{
            key_expr: "storage/mem1/**",
            strip_prefix: "storage/mem1",
            volume: "memory",
            replication: {
                interval: 2,
            }
        }"#,
    )?;
    Ok(cfg)
}

pub async fn open(
    cfg: zenoh::Config,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let namespace: [u8; 8] = {
        blake3::hash(namespace.as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("8 is equal to 8")
    };
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    if let Ok(peers) = env::var("EXO_CONNECT_PEERS") {
        for peer in peers
            .split(",")
            .map(str::trim)
            .filter(|peer| !peer.is_empty())
        {
            let Some((zid, addr)) = peer.split_once("@") else {
                log::warn!("ignoring EXO_CONNECT_PEERS entry without @: {peer}");
                continue;
            };
            let Ok(zid) = ZenohId::from_str(zid.trim()).inspect_err(|e| {
                log::warn!("failed to parse EXO_CONNECT_PEERS zid {zid}: {e}");
            }) else {
                continue;
            };
            let addr = addr.trim();
            let locator_result = if addr.contains("/") {
                Locator::from_str(addr)
            } else {
                Locator::new("tcp", addr.to_string(), "")
            };
            let Ok(locator) = locator_result.inspect_err(|e| {
                log::warn!("failed to parse EXO_CONNECT_PEERS locator {addr}: {e}");
            }) else {
                continue;
            };
            log::warn!("EXO_CONNECT_PEERS connecting to {zid} at {locator}");
            let connected = runtime.connect_peer(&zid.into(), &[locator]).await;
            log::warn!("EXO_CONNECT_PEERS connect_peer({zid}) -> {connected}");
        }
    }
    let mut discovery =
        Discovery::new(z.zid(), namespace, listen_port, discovery_service_port).await?;
    let _jh = Arc::new(AbortOnDrop(tokio::task::spawn(async move {
        loop {
            let Ok(discovered) = discovery.next().await.inspect_err(|e| {
                log::warn!("discovery error {e}");
            }) else {
                continue;
            };

            if discovered.zid > runtime.zid() {
                log::debug!("not connecting to peer with greater zid");
                continue;
            }

            let Ok(locator) =
                Locator::new("tcp", discovered.addr.to_string(), "").inspect_err(|e| {
                    log::warn!("failed to parse locator from addr: {e}");
                })
            else {
                continue;
            };

            runtime
                .connect_peer(&discovered.zid.into(), &[locator])
                .await;
        }
    })));
    Ok(Session { z, _jh })
}

struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Clone)]
pub struct Session {
    pub z: ZSession,
    _jh: Arc<AbortOnDrop>,
}
