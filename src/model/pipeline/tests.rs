use super::*;
use std::sync::{Arc, Barrier};

#[test]
fn shared_state_is_ordered_while_different_layers_overlap() -> Result<()> {
    let layers: Layers<Vec<usize>> = vec![vec![], vec![], vec![]].into();
    let overlap = Arc::new(Barrier::new(2));
    run(
        &layers,
        None,
        |i, state| {
            state.push(0);
            if i == 1 {
                overlap.wait();
            }
            Ok(())
        },
        |i, state| {
            if i == 0 {
                overlap.wait();
            }
            state.push(1);
            Ok(())
        },
    )?;
    assert!(layers.iter().all(|state| *state == [0, 1]));
    Ok(())
}

#[test]
fn errors_on_either_lane_release_the_peer() {
    for failing_lane in [0, 1] {
        let layers: Layers<()> = vec![(), (), ()].into();
        let result = run(
            &layers,
            None,
            |_, _| {
                ensure!(failing_lane != 0, "leader failure");
                Ok(())
            },
            |_, _| {
                ensure!(failing_lane != 1, "follower failure");
                Ok(())
            },
        );
        assert!(result.is_err());
    }
}

#[test]
fn cancellation_drains_the_pipeline_before_returning() {
    let layers: Layers<usize> = vec![0, 0, 0].into();
    let cancelled = AtomicBool::new(false);
    assert!(
        run(
            &layers,
            Some(&cancelled),
            |_, state| {
                *state += 1;
                cancelled.store(true, Ordering::Relaxed);
                Ok(())
            },
            |_, state| {
                *state += 1;
                Ok(())
            }
        )
        .is_err()
    );
    assert_eq!(layers.iter().map(|state| *state).sum::<usize>(), 1);
}
