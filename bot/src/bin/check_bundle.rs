use bot::catalog::Bundle;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(directory) = args.next() else {
        eprintln!("Usage: check_bundle <directory>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("Usage: check_bundle <directory>");
        std::process::exit(2);
    }
    match Bundle::load(directory) {
        Ok(bundle) => println!("{} records={}", bundle.identity(), bundle.catalog().len()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
