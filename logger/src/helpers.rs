pub struct PrettyVec<T>(pub Vec<T>);

impl<T: AsRef<str>> std::fmt::Debug for PrettyVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .0
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "[{}]", joined)
    }
}
