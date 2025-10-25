use once_cell::sync::OnceCell;
use tracing::Subscriber;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};

static TRACING: OnceCell<()> = OnceCell::new();

pub fn init_tracing() {
    TRACING.get_or_init(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt_layer = fmt::layer().with_target(false).with_ansi(false);
        let subscriber = Registry::default().with(filter).with(fmt_layer);
        tracing::subscriber::set_global_default(subscriber).expect("set global subscriber");
    });
}

pub fn subscriber() -> impl Subscriber {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    Registry::default().with(filter).with(fmt::layer())
}
