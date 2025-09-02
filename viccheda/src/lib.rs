#![allow(unused)]

mod rules;
mod split;

#[cfg(test)]
pub(crate) fn init_logger(subject: &'static str) -> once_cell::sync::OnceCell<logger::Logger> {
    use env_logger;
    use logger::Logger;
    use once_cell::sync::OnceCell;

    static INIT: OnceCell<Logger> = OnceCell::new();

    INIT.get_or_init(|| {
        let _ = env_logger::builder().is_test(true).try_init();
        Logger::new(true, subject)
    });

    INIT.clone()
}
