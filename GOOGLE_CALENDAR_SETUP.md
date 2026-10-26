# Google Calendar connection

Hermes currently requests read-only access to events on the signed-in user's primary calendar. It can list upcoming events; it cannot create, edit, share, or delete calendar data.

1. In Google Cloud Console, create/select a project, enable **Google Calendar API**, and configure the OAuth consent screen. If the app is in testing mode, add the Google account you will use as a test user.
2. Create an OAuth client ID of type **Web application**. Add this exact authorized redirect URI:

   `http://localhost:5000/google-calendar/callback`

3. Add the client ID and client secret directly to the private `env` file:

   ```env
   GOOGLE_CLIENT_ID=your-client-id
   GOOGLE_CLIENT_SECRET=your-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:5000/google-calendar/callback
   ```

4. Restart Hermes, open Configuration in its browser page, and choose **Connect with Google**. Review Google's consent page and grant the requested read-only event scope.

Hermes stores the OAuth token in `google_calendar_token.json` with owner-only file permissions. That file is ignored by Git. To revoke access, remove Hermes from your Google Account's third-party connections and delete this token file.
