//! Completion-gated query readback and visibility updates.

use super::*;

impl<B: OcclusionWorkloadBackend> Workload<B> {
    pub fn consume(
        &mut self,
        d: &Device<B>,
        ticket: ReadbackTicket<B>,
        c: &CompletionPoint<B>,
    ) -> Result<()> {
        utils::block_on(d.wait_completion(c))?;
        // SAFETY: `submit` returned this end checkpoint for the command step
        // that created `ticket`, and the wait above observed that exact point.
        let v = utils::block_on(unsafe { d.read_after(ticket, c) })?.bytes;
        self.consume_bytes(&v)
    }
    pub(crate) async fn consume_async(
        &mut self,
        d: &Device<B>,
        ticket: ReadbackTicket<B>,
        c: &CompletionPoint<B>,
    ) -> Result<()> {
        d.wait_completion(c).await?;
        // SAFETY: this ticket and checkpoint come from one `Frame`; the
        // awaited end checkpoint covers that frame's query resolve.
        let readback = unsafe { d.read_after(ticket, c) }.await?;
        self.consume_bytes(&readback.bytes)
    }
    fn consume_bytes(&mut self, v: &[u8]) -> Result<()> {
        if v.len() != 16 {
            return Err(Error::Backend("occlusion result size".into()));
        }
        let passed = [
            u64::from_le_bytes(v[..8].try_into().map_err(|_| Error::InvalidState)?),
            u64::from_le_bytes(v[8..].try_into().map_err(|_| Error::InvalidState)?),
        ];
        if std::env::var_os("FLUXEL_QUERY_TRACE").is_some() {
            println!(
                "occlusion results: teapot={} sphere={}",
                passed[0], passed[1]
            );
        }
        self.visible = [passed[0] != 0, passed[1] != 0];
        Ok(())
    }
}
