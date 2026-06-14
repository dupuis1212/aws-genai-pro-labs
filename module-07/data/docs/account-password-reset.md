---
title: Resetting your password and securing your account
category: account
---

# Resetting your password and securing your account

Account access problems are the fastest tickets to resolve once you know which of
the three states an account is in: the password is simply forgotten, the account
is locked after too many attempts, or two-factor authentication is blocking
sign-in. Each has a different fix.

## Resetting a forgotten password

1. On the sign-in page, click **Forgot password?**.
2. Enter the email on the account. CloudCart sends a reset link that is valid for
   one hour.
3. Open the link and set a new password. It must be at least 12 characters with
   one number and one symbol.
4. The reset signs out every other session, so anyone using a shared login has to
   sign in again.

If the reset email does not arrive within a few minutes, check spam, confirm the
email is the one on the account, and make sure your mail server is not blocking
the CloudCart sending domain.

## A locked account

After five failed sign-in attempts, an account is locked for 30 minutes as a
brute-force protection. Waiting it out clears the lock automatically; there is no
need to contact support. A password reset also clears the lock immediately.

## Two-factor authentication problems

If 2FA codes are not working, the device clock is usually out of sync, which
breaks time-based codes. Sync the device clock and try again. If the
authenticator device is lost, use a saved backup code, or have an account owner
remove 2FA from **Settings -> Security -> Two-factor authentication** so you can
re-enroll.

## Changing the account owner email

Only an account owner can change the owner email, from **Settings -> Account**.
The change requires confirming a code sent to BOTH the old and the new address, so
a compromised inbox alone cannot move the account.
